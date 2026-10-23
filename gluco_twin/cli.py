"""GlucoTwin Command Line Interface."""
import argparse
import sys
import torch
from gluco_twin.config import cfg, CURRENT_MODEL_DIR, PROCESSED_DATA_DIR
from gluco_twin.data.dataset import prepare_and_cache_datasets, TwinDataset
from gluco_twin.models.train import train_fusion_twin_net
from gluco_twin.eval.evaluate import run_full_evaluation
from gluco_twin.models.fusion_net import FusionTwinNet
from gluco_twin.models.explain import TwinExplainer
from gluco_twin.models.whatif import WhatIfSimulator

def main():
    parser = argparse.ArgumentParser(description="GlucoTwin Digital Twin CLI")
    subparsers = parser.add_subparsers(dest="command")

    # data
    subparsers.add_parser("data", help="Preprocess datasets and cache sliding windows")

    # train
    train_parser = subparsers.add_parser("train", help="Train FusionTwinNet")
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--lr", type=float, default=1e-3)

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate model on held-out test cohort")

    # explain
    explain_parser = subparsers.add_parser("explain", help="Explain spike risk factors")
    explain_parser.add_argument("--patient-id", type=str, default="P086")

    # whatif
    whatif_parser = subparsers.add_parser("whatif", help="Run counterfactual what-if simulation")
    whatif_parser.add_argument("--patient-id", type=str, default="P086")
    whatif_parser.add_argument("--carbs", type=float, default=45.0)
    whatif_parser.add_argument("--steps", type=float, default=2000.0)

    args = parser.parse_args()

    if args.command == "data":
        prepare_and_cache_datasets()
    elif args.command == "train":
        train_fusion_twin_net(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.command == "evaluate":
        run_full_evaluation()
    elif args.command == "explain":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = FusionTwinNet().to(device)
        model.load_state_dict(torch.load(CURRENT_MODEL_DIR / "model.pt", map_location=device))
        test_ds = TwinDataset(str(PROCESSED_DATA_DIR / "test.npz"))
        explainer = TwinExplainer(model, device)
        item = test_ds[0]
        exp = explainer.explain_prediction(
            item["history"].unsqueeze(0),
            item["covariates"].unsqueeze(0),
            item["static"].unsqueeze(0),
        )
        print("\n=== Clinical Explainability Breakdown ===")
        print(f"Base Predicted Spike Risk: {exp['base_prob']*100:.1f}%")
        print("\nFactor Attribution:")
        for factor, pct in exp["sorted_factors"]:
            print(f"  - {factor:<25}: {pct:>5.1f}%")
        print(f"\nNarrative: {exp['narrative']}")
    elif args.command == "whatif":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        model = FusionTwinNet().to(device)
        model.load_state_dict(torch.load(CURRENT_MODEL_DIR / "model.pt", map_location=device))
        test_ds = TwinDataset(str(PROCESSED_DATA_DIR / "test.npz"))
        sim = WhatIfSimulator(model, device)
        item = test_ds[0]
        base_res = sim.simulate(item["history"].unsqueeze(0), item["covariates"].unsqueeze(0), item["static"].unsqueeze(0))
        whatif_res = sim.simulate(
            item["history"].unsqueeze(0),
            item["covariates"].unsqueeze(0),
            item["static"].unsqueeze(0),
            add_carbs_g=args.carbs,
            add_steps=args.steps,
        )
        print("\n=== What-If Intervention Simulation ===")
        print(f"Base Peak Glucose:     {base_res['peak_glucose']:.1f} mg/dL | Event Prob: {base_res['event_prob']*100:.1f}%")
        print(f"Intervention (+{args.carbs}g carbs, +{args.steps} steps):")
        print(f"Intervention Peak:     {whatif_res['peak_glucose']:.1f} mg/dL | Event Prob: {whatif_res['event_prob']*100:.1f}%")
        delta = whatif_res['peak_glucose'] - base_res['peak_glucose']
        print(f"Delta Peak Glucose:    {delta:+.1f} mg/dL")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
