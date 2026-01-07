import sys
import logging
import traceback

# Setup logging immediately
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

try:
    import os
    import runpod
    from utils import JobInput
    from engine import vLLMEngine, OpenAIvLLMEngine

    logger.info("Initializing vLLM Engine...")
    vllm_engine = vLLMEngine()
    logger.info("vLLM Engine initialized successfully.")

    logger.info("Initializing OpenAI Engine...")
    openai_engine = OpenAIvLLMEngine(vllm_engine)
    logger.info("OpenAI Engine initialized successfully.")

except Exception as e:
    logger.critical(f"Fatal error during initialization: {e}")
    traceback.print_exc()
    sys.exit(1)


async def handler(job):
    job_input = JobInput(job["input"])
    engine = openai_engine if job_input.openai_route else vllm_engine
    results_generator = engine.generate(job_input)
    async for batch in results_generator:
        yield batch


runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
