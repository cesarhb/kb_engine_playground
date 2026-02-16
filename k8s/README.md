# Kubernetes (MicroK8s)

- **ConfigMap** `configmap-doc-sources.yaml`: doc sources for the ingestion pipeline. Keeps `config/doc_sources.yaml` in sync or customize for K8s.
- **CronJob** `cronjob-ingestion.yaml`: runs the ingestion pipeline on a schedule (default 02:00 daily). Replace the image with your built image (e.g. push to a registry and set `image`).
- **Secret** `secret.example.yaml`: copy to `secret.yaml`, fill in `database-url` (and optional API keys), then `kubectl apply -f k8s/secret.yaml`. Do not commit `secret.yaml`.

## Apply

```bash
kubectl apply -f k8s/configmap-doc-sources.yaml
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/cronjob-ingestion.yaml
```

For local MicroK8s: alias `kubectl` to `microk8s kubectl` or use `microk8s kubectl apply -f k8s/`.
