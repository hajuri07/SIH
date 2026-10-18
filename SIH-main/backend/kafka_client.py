"""
Kafka Producer & Consumer Integration for VayuPulse Telemetry Streaming.
Supports production Apache Kafka with automatic graceful fallback to local direct mode
when a Kafka broker is not accessible.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer

from .config import settings
from .retry import async_retry

logger = logging.getLogger("vayupulse.kafka")


class KafkaStreamingManager:
    def __init__(self):
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumer: Optional[AIOKafkaConsumer] = None
        self.is_connected: bool = False
        self._consumer_task: Optional[asyncio.Task] = None
        self.message_handler: Optional[Callable[[Dict[str, Any]], Any]] = None

    async def start(self):
        """Attempts to initialize Kafka producer with timeout."""
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                request_timeout_ms=2000
            )
            # Short timeout to avoid blocking startup if broker is offline
            await asyncio.wait_for(self.producer.start(), timeout=2.5)
            self.is_connected = True
            logger.info(f"Kafka producer connected to {settings.KAFKA_BOOTSTRAP_SERVERS}")
            print(f"[INFO] Kafka producer successfully connected to {settings.KAFKA_BOOTSTRAP_SERVERS}")
        except Exception as e:
            self.is_connected = False
            self.producer = None
            logger.info(f"Kafka broker unavailable ({e}). Using direct local processing.")
            print(f"[INFO] Kafka broker not available at {settings.KAFKA_BOOTSTRAP_SERVERS}. Operating in direct in-memory streaming mode.")

    @async_retry(max_retries=2, initial_delay=0.05, backoff_factor=2.0)
    async def send_telemetry(self, packet: Dict[str, Any]) -> bool:
        """Publishes telemetry packet to Kafka topic, or dispatches directly to consumer in local fallback mode."""
        if self.is_connected and self.producer:
            try:
                await self.producer.send_and_wait(settings.KAFKA_TOPIC_TELEMETRY, packet)
                return True
            except Exception as e:
                logger.warning(f"Failed to emit telemetry to Kafka: {e}")
                return False
        else:
            # Direct fallback: if a message handler is registered, dispatch to it asynchronously
            if self.message_handler:
                try:
                    asyncio.create_task(self.message_handler(packet))
                    return True
                except Exception as e:
                    logger.warning(f"Direct telemetry dispatch error: {e}")
            return False

    @async_retry(max_retries=2, initial_delay=0.05, backoff_factor=2.0)
    async def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Publishes anomaly alert to Kafka topic."""
        if self.is_connected and self.producer:
            try:
                await self.producer.send_and_wait(settings.KAFKA_TOPIC_ALERTS, alert)
                return True
            except Exception as e:
                logger.warning(f"Failed to emit alert to Kafka: {e}")
                return False
        return False

    def register_handler(self, handler: Callable[[Dict[str, Any]], Any]):
        """Registers the telemetry inference message handler."""
        self.message_handler = handler

    async def start_consumer(self, message_handler: Callable[[Dict[str, Any]], Any]):
        """Starts background Kafka consumer loop if connected, or registers handler for direct local dispatch."""
        self.message_handler = message_handler
        if not self.is_connected:
            logger.info("Kafka consumer registered in direct local streaming mode.")
            return

        async def _consumer_loop():
            try:
                consumer = AIOKafkaConsumer(
                    settings.KAFKA_TOPIC_TELEMETRY,
                    bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
                    group_id="vayupulse-prognostics-group",
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    auto_offset_reset="latest"
                )
                await consumer.start()
                self.consumer = consumer
                logger.info("Kafka consumer started listening on " + settings.KAFKA_TOPIC_TELEMETRY)
                async for msg in consumer:
                    try:
                        await message_handler(msg.value)
                    except Exception as handler_err:
                        logger.error(f"Error handling Kafka message: {handler_err}")
            except Exception as e:
                logger.warning(f"Kafka consumer stopped: {e}")
            finally:
                if self.consumer:
                    await self.consumer.stop()

        self._consumer_task = asyncio.create_task(_consumer_loop())

    def get_readiness(self) -> Dict[str, Any]:
        """Reports actual Kafka connectivity, consumer, and streaming readiness."""
        consumer_active = (
            (self._consumer_task is not None and not self._consumer_task.done())
            if self.is_connected
            else (self.message_handler is not None)
        )
        return {
            "ready": self.is_connected,
            "connected": self.is_connected,
            "broker": settings.KAFKA_BOOTSTRAP_SERVERS,
            "producer_ready": self.producer is not None,
            "consumer_running": consumer_active,
            "mode": "broker" if self.is_connected else "direct_fallback"
        }

    async def stop(self):
        """Gracefully stops producer and consumer."""
        if self._consumer_task and not self._consumer_task.done():
            self._consumer_task.cancel()
        if self.consumer:
            try:
                await self.consumer.stop()
            except Exception:
                pass
        if self.producer:
            try:
                await self.producer.stop()
            except Exception:
                pass
        self.is_connected = False


kafka_manager = KafkaStreamingManager()
