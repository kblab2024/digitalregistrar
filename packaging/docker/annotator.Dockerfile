# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS builder
WORKDIR /build
COPY vendor/ vendor/
COPY pyproject.toml ./
COPY src/ src/
COPY apps/annotator/ apps/annotator/
RUN pip install --no-cache-dir --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir /wheels vendor/tnmhelper-0.1.0-py3-none-any.whl && \
    pip wheel --no-cache-dir --wheel-dir /wheels . && \
    pip wheel --no-cache-dir --wheel-dir /wheels apps/annotator

FROM python:3.11-slim AS runtime
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
ENV STREAMLIT_SERVER_PORT=8500 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0
EXPOSE 8500
CMD ["registrar-annotate-dummy", "--no-browser"]
