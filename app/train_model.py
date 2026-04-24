from app.services.modeling import train_model_bundle


def main() -> None:
    bundle = train_model_bundle(refresh_data=True)
    print(f"Best model: {bundle.model_name}")
    print(f"Samples: {bundle.sample_count}")
    for model_name, metrics in bundle.metrics.items():
        print(
            f"{model_name}: log_loss={metrics['log_loss']:.4f}, "
            f"accuracy={metrics['accuracy']:.4f}, roc_auc={metrics['roc_auc']:.4f}"
        )


if __name__ == "__main__":
    main()
