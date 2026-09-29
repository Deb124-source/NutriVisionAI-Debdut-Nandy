# NutriVision AI — deployable mixed-meal MVP

**Real inference, not a fake demo:** FastAPI + OWLv2 zero-shot food-region bounding boxes + CLIP crop classification. Users must confirm serving sizes; there is **no validated automatic mass or pixel-accurate segmentation**. Nutrition uses approximate recipe averages per 100 g and is not medical advice.

## Local

Requires Python 3.11 and several GB of RAM. First request downloads two public models (several GB total, depending on versions); it may take minutes and require persistent cache.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Visit http://localhost:7860. `/health` does not load the models. `/docs` provides API documentation. Test `POST /api/analyze` with an actual photo, and `POST /api/nutrition` with confirmed weights.

## Deploy

**Hugging Face Spaces (Docker):** Create a new Docker Space, upload all project files at repository root, set the Space's `app_port` to `7860` in its README YAML front matter if required, and use adequate CPU/RAM. Space URL will be `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE` and app URL `https://YOUR_USERNAME-YOUR_SPACE.hf.space` after successful deployment. Actual URL depends on your account and Space name.

**Render:** Push files to a GitHub repository, choose New > Blueprint, and connect the repo containing `render.yaml`. The Starter plan is suggested because both models are too heavy for typical free-tier memory. Render provides the actual URL after deployment.

## Limits

- OWLv2 predicts *boxes*, not pixel segmentation. Adjacent foods, sauces and overlap can confuse it.
- CLIP scores are not calibrated food probabilities. Optional text description is collected but not fused into model inference in this MVP.
- Plate geometry, depth, density, grams and calorie estimates are **not** inferred from pixels. Enter grams manually. No unsupported claims of automatic portion measurement.
- Recipe averages vary by preparation. Add verified regional food data and measured-weight training samples before treating this as research-grade portion estimation.
- Model inference on CPU is slow; cold starts can be lengthy. Do not upload private photos to public third-party Spaces without reviewing their privacy settings.
