import logging
import os
from typing import Annotated, Any

import structlog
from fastapi import Body, FastAPI, Query
from fastapi.responses import Response
from mangum import Mangum

from poprox_concepts.api.recommendations.v2 import ProtocolModelV2_0, RecommendationRequestV2, RecommendationResponseV2
from poprox_recommender.api.gzip import GzipRoute

logger = logging.getLogger(__name__)

app = FastAPI()
app.router.route_class = GzipRoute


logger = logging.getLogger(__name__)


@app.get("/warmup")
def warmup(response: Response):
    # Headers set on the response param get included in the response wrapped around return val
    response.headers["poprox-protocol-version"] = ProtocolModelV2_0().protocol_version.value

    # Pre-warm persona_recommender pipeline to avoid timeout on first use
    from poprox_recommender.recommenders import discover_pipelines, get_pipeline

    available_pipeline_names = discover_pipelines()

    # Pre-load persona_recommender to cache the models
    try:
        logger.info("Pre-warming persona_recommender pipeline...")
        get_pipeline("persona_recommender")  # Just cache it, don't need to store
        logger.info("✅ persona_recommender pre-warmed successfully")
    except Exception as e:
        logger.warning(f"❌ Failed to pre-warm persona_recommender: {e}")

    return available_pipeline_names


@app.post("/")
def root(
    body: Annotated[dict[str, Any], Body()],
    pipeline: Annotated[str | None, Query()] = None,
):
    logger.info(f"Decoded body: {body}")
    logger.info(f"Pipeline query parameter: {pipeline}")

    # Also check for pipeline in request body as fallback
    body_pipeline = body.get("pipeline") if isinstance(body, dict) else None
    if body_pipeline:
        logger.info(f"Pipeline from request body: {body_pipeline}")
        pipeline = body_pipeline
    else:
        body_pipeline = None

    req = RecommendationRequestV2.model_validate(body)

    candidate_articles = req.candidates.articles
    num_candidates = len(candidate_articles)

    if num_candidates < req.num_recs:
        msg = f"Received insufficient candidates ({num_candidates}) in a request for {req.num_recs} recommendations."
        raise ValueError(msg)

    logger.info(f"Selecting articles from {num_candidates} candidates...")

    # Lazy import heavy dependencies only when needed
    from poprox_recommender.recommenders import select_articles
    from poprox_recommender.topics import user_locality_preference, user_topic_preference

    profile = req.interest_profile
    profile.click_topic_counts = user_topic_preference(req.interacted.articles, profile.click_history)
    profile.click_locality_counts = user_locality_preference(req.interacted.articles, profile.click_history)

    # TEMPORARY DEBUG: Force persona_recommender to test
    logger.info(f"PIPELINE DEBUG - Received: query='{pipeline}', body='{body_pipeline}', final='{pipeline}'")

    # BULLETPROOF: Always use persona_recommender (default is now set to persona_recommender)
    selected_pipeline = pipeline or "persona_recommender"
    logger.info(f"🎯 SELECTED PIPELINE: {selected_pipeline}")

    try:
        logger.info(f"🔥 ATTEMPTING TO USE PIPELINE: {selected_pipeline}")
        outputs = select_articles(
            req.candidates,
            req.interacted,
            profile,
            {"pipeline": selected_pipeline},
        )
        logger.info(f"✅ SUCCESS: Pipeline {selected_pipeline} executed successfully")
        actual_pipeline_used = selected_pipeline
    except Exception as e:
        logger.error(f"❌ PIPELINE FAILED: {selected_pipeline} failed with error: {e}")
        logger.error("Falling back to nrms_topic_scores...")
        try:
            outputs = select_articles(
                req.candidates,
                req.interacted,
                profile,
                {"pipeline": "nrms_topic_scores"},
            )
            actual_pipeline_used = "nrms_topic_scores"
            logger.info("✅ FALLBACK SUCCESS: nrms_topic_scores worked")
        except Exception as fallback_error:
            logger.error(f"❌ EVEN FALLBACK FAILED: {fallback_error}")
            raise

    # Build persona from last 50 clicks using Gemini
    def _extract_click_texts() -> list[str]:
        try:
            clicks = getattr(req, "interacted", None)
            if clicks and hasattr(clicks, "articles"):
                click_articles = clicks.articles or []
            else:
                click_articles = []
        except Exception as e:
            logger.warning("Failed to extract click articles", exc_info=e)
            click_articles = []

        logger.info(f"Found {len(click_articles)} click articles for persona generation")

        texts: list[str] = []
        # Use last 50 articles for balanced detail and speed
        for i, a in enumerate(click_articles[:50]):
            # Get only the title/headline - no summaries for faster processing
            title = getattr(a, "headline", None) or getattr(a, "title", None)

            # Debug: log article structure for first few articles
            if i < 3:
                logger.info(f"Article {i} - title: {title}, type: {type(a)}")

            # Only add titles (skip articles without titles)
            if title:
                texts.append(title)

        logger.info(f"Extracted {len(texts)} text snippets for persona generation")
        if texts:
            logger.info(f"Sample extracted text: {texts[0][:200]}..." if len(texts[0]) > 200 else texts[0])
        return texts

    def _generate_persona_with_gemini(texts: list[str]) -> str | None:
        # Lazy import Gemini only when needed for persona generation
        import google.generativeai as genai

        # Get API key from environment variable
        api_key = os.environ.get("GEMINI_API_KEY")

        if not api_key:
            logger.error("GEMINI_API_KEY environment variable not set")
            return None

        logger.info("Using Gemini API key from GEMINI_API_KEY environment variable")

        logger.info(f"Starting Gemini persona generation with {len(texts)} texts")
        if not texts:
            logger.warning("No texts available for persona generation")
            return None

        # Try multiple Gemini models in order of speed (fastest first for Lambda timeout)
        models_to_try = [
            "gemini-1.5-flash",  # Fastest model
            "gemini-2.0-flash-exp",  # Fast experimental model
            "gemini-1.0-pro",  # Stable fallback
            "gemini-1.5-pro",  # Slowest but highest quality
        ]

        for model_name in models_to_try:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                # Join titles with simple numbering for clarity
                numbered_titles = [f"{i + 1}. {title}" for i, title in enumerate(texts)]
                click_block = "\n".join(numbered_titles)
                logger.info(f"Generated click block with {len(click_block)} characters from {len(texts)} titles")

                # Focused prompt for interest-based persona without demographics
                prompt = (
                    "Analyze these clicked news article titles to describe this user's reading preferences. "
                    "Focus on: main topic interests, fine-grained sub-topical interests, preferred geographic regions,"
                    "depth vs breadth preferences, news consumption motivations, and content style. "
                    "Write 4-7 sentences about their interests and reading patterns,"
                    "that can be used to generate recommendations."
                    "Do NOT invent demographics, names, or personal details. "
                    "Refer to them as 'this user' or 'the reader'.\n\n" + click_block
                )
                logger.info(f"Sending request to Gemini API with {model_name}...")

                # Generate with timeout awareness - fail fast if approaching Lambda limit
                import time

                start_time = time.time()
                resp = model.generate_content(prompt)
                end_time = time.time()

                logger.info(f"Received response from Gemini {model_name} in {end_time - start_time:.2f}s: {type(resp)}")

                text = getattr(resp, "text", None)
                if text:
                    logger.info(f"Successfully extracted text from {model_name} resp.text: {len(text)} characters")
                    return text.strip()

                logger.info(f"No text in resp.text for {model_name}, trying candidates...")
                # Some SDK versions nest candidates
                if hasattr(resp, "candidates") and resp.candidates:
                    logger.info(f"Found {len(resp.candidates)} candidates")
                    for i, c in enumerate(resp.candidates):
                        logger.info(f"Candidate {i}: {type(c)}")
                        content = getattr(c, "content", None)
                        if content:
                            parts = getattr(content, "parts", None)
                            if parts:
                                logger.info(f"Found {len(parts)} parts in candidate {i}")
                                out = "".join([getattr(p, "text", "") for p in parts]).strip()
                                if out:
                                    logger.info(
                                        f"Successfully extracted text from {model_name} candidate {i}: "
                                        f"{len(out)} characters"
                                    )
                                    return out

                logger.warning(f"No text found in response from {model_name} - response structure might have changed")
                continue  # Try next model

            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini {model_name} failed: {error_msg}")

                # Check if it's a quota/rate limit error
                is_quota_error = any(term in error_msg.lower() for term in ["quota", "429", "rate limit", "exceeded"])
                is_auth_error = any(
                    term in error_msg.lower() for term in ["api key", "authentication", "unauthorized", "403"]
                )

                if is_quota_error:
                    logger.info(f"Quota/rate limit hit for {model_name}, trying next model...")
                    continue  # Try next model
                elif is_auth_error:
                    logger.error(f"Authentication error with {model_name}: {error_msg} - check API key")
                    continue  # Try next model
                else:
                    logger.error(f"Non-quota error with {model_name}: {e}")
                    continue  # Try next model anyway

        logger.error("All Gemini models failed")
        return None

    # Check if persona generation is enabled (can be disabled via env var for debugging)
    persona_enabled = os.environ.get("POPROX_DISABLE_PERSONA", "").lower() not in ("true", "1", "yes")

    if persona_enabled:
        # First, let's test with a simple hardcoded persona to verify the pipeline works
        test_mode = os.environ.get("POPROX_TEST_PERSONA", "").lower() in ("true", "1", "yes")

        if test_mode:
            logger.info("TEST MODE: Using hardcoded persona")
            generated_persona = (
                "TEST PERSONA: This user shows interest in technology and science news, "
                "with a preference for in-depth analysis and emerging trends."
            )
        else:
            click_texts = _extract_click_texts()
            logger.info(f"Attempting persona generation with {len(click_texts)} click texts")

            if len(click_texts) == 0:
                logger.warning("No click texts extracted - check article field names")
                generated_persona = None
            else:
                generated_persona = _generate_persona_with_gemini(click_texts)

        if generated_persona:
            logger.info(f"Successfully generated persona: {generated_persona[:100]}...")
        else:
            logger.warning("Persona generation failed - using default message")
    else:
        logger.info("Persona generation disabled via POPROX_DISABLE_PERSONA environment variable")
        generated_persona = None

    # Inject persona into existing metadata field to surface in newsletter header without cross-repo changes
    meta = outputs.meta.model_dump()

    # Update the name to show the actual pipeline used
    meta["name"] = actual_pipeline_used

    # Only add persona if we actually used persona_recommender
    if actual_pipeline_used == "persona_recommender":
        persona = generated_persona or "persona not available"
        meta["version"] = (f'{meta.get("version", "")} | persona="{persona}"').strip()
    else:
        logger.warning(f"Not adding persona because actual pipeline was: {actual_pipeline_used}")

    resp_body = RecommendationResponseV2.model_validate({"recommendations": outputs.default, "recommender": meta})

    logger.info(f"Response body: {resp_body}")
    return resp_body.model_dump()


handler = Mangum(app)


if "AWS_LAMBDA_FUNCTION_NAME" in os.environ and not structlog.is_configured():
    # Serverless doesn't set up logging like the AWS Lambda runtime does, so we
    # need to configure base logging ourselves. The AWS_LAMBDA_RUNTIME_API
    # environment variable is set in a real runtime environment but not the
    # local Serverless run, so we can check for that.  We will log at DEBUG
    # level for local testing.
    if "AWS_LAMBDA_RUNTIME_API" not in os.environ:
        logging.basicConfig(level=logging.DEBUG)
        # make sure we have debug for all of our code
        logging.getLogger("poprox_recommender").setLevel(logging.DEBUG)
        logger.info("local logging enabled")

    # set up structlog to dump to standard logging
    # TODO: enable JSON logs
    structlog.configure(
        [
            structlog.processors.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.MaybeTimeStamper(),
            structlog.processors.KeyValueRenderer(key_order=["event", "timestamp"]),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    structlog.stdlib.get_logger(__name__).info(
        "structured logging initialized",
        function=os.environ["AWS_LAMBDA_FUNCTION_NAME"],
        region=os.environ.get("AWS_REGION", None),
    )
