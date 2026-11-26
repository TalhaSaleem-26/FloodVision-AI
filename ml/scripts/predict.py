import argparse
import json
import sys
import uuid

from pathlib import Path


ML_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ML_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from floodvision.inference.pipeline import (
    FloodInferencePipeline
)

from floodvision.inference.pipeline_config import (
    PipelineConfig
)

from floodvision.inference.exceptions import (
    InferenceError
)


def build_parser():

    parser = argparse.ArgumentParser(
        prog="floodvision-predict",
        description=(
            "FloodVision AI - Satellite water "
            "segmentation and geospatial analytics"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input two-band SAR GeoTIFF"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output analysis directory"
    )

    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=ML_DIR / "artifacts" / "production",
        help="Production model bundle directory"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0"
    )

    parser.add_argument(
        "--storage",
        choices=["auto", "memory", "disk"],
        default="auto"
    )

    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=128
    )

    parser.add_argument(
        "--memory-budget-mib",
        type=int,
        default=128
    )

    parser.add_argument(
        "--max-pixels",
        type=int,
        default=100_000_000
    )

    parser.add_argument(
        "--max-tiles",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--max-vector-pixels",
        type=int,
        default=4_000_000
    )

    parser.add_argument(
        "--max-regions",
        type=int,
        default=100_000
    )

    parser.add_argument(
        "--connectivity",
        type=int,
        choices=[4, 8],
        default=4
    )

    parser.add_argument(
        "--simplification-tolerance",
        type=float,
        default=5.0
    )

    parser.add_argument(
        "--no-rasters",
        action="store_true"
    )

    parser.add_argument(
        "--no-vectors",
        action="store_true"
    )

    parser.add_argument(
        "--no-area",
        action="store_true"
    )

    parser.add_argument(
        "--inference-only",
        action="store_true"
    )

    parser.add_argument(
        "--allow-unverified-band-order",
        action="store_true",
        help=(
            "Allow missing band descriptions only "
            "when VV/VH order is independently verified"
        )
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Print complete JSON analysis result"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show full traceback on failure"
    )

    return parser


def build_config(args):

    return PipelineConfig(
        device=args.device,
        storage=args.storage,
        work_dir=args.work_dir,
        overlap=args.overlap,
        memory_budget_mib=args.memory_budget_mib,
        max_pixels=args.max_pixels,
        max_tiles=args.max_tiles,
        max_vector_pixels=args.max_vector_pixels,
        max_regions=args.max_regions,
        connectivity=args.connectivity,
        simplification_tolerance_m=(
            args.simplification_tolerance
        ),
        allow_unverified_band_order=(
            args.allow_unverified_band_order
        )
    )


def resolve_output_dir(args):

    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()

    return (
        ML_DIR /
        "outputs" /
        f"analysis_{uuid.uuid4().hex}"
    )


def print_summary(result):

    statistics = result.statistics

    print("=" * 65)
    print("FLOODVISION AI")
    print("Satellite Water Segmentation")
    print("=" * 65)

    print(f"Analysis ID: {result.analysis_id}")
    print(f"Status: {result.status}")
    print(f"Source: {result.source_file}")
    print(f"Model version: {result.model_version}")
    print(f"Device: {result.device}")
    print(f"Threshold: {result.threshold}")

    print("\nPIXEL STATISTICS")
    print("-" * 65)

    print(
        f"Total pixels: "
        f"{statistics['total_pixels']:,}"
    )

    print(
        f"Valid pixels: "
        f"{statistics['valid_pixels']:,}"
    )

    print(
        f"Invalid pixels: "
        f"{statistics['invalid_pixels']:,}"
    )

    print(
        f"Predicted water pixels: "
        f"{statistics['water_pixels']:,}"
    )

    coverage = result.water_coverage

    if coverage is None:
        print("Water coverage: N/A")

    else:
        print(f"Water coverage: {coverage:.4f}%")

    if result.area_analytics is not None:

        print("\nAREA ANALYTICS")
        print("-" * 65)

        if statistics["valid_pixels"] == 0:

            print(
                "Water area: N/A "
                "(no valid satellite data)"
            )

        else:

            print(
                f"Predicted water area: "
                f"{result.water_area_km2:.6f} km²"
            )

    if result.vectorization is not None:

        print("\nVECTOR ANALYTICS")
        print("-" * 65)

        print(
            "Water regions:",
            result.vectorization["region_count"]
        )

    print("\nTILING")
    print("-" * 65)

    for key, value in result.tiling.items():
        print(f"{key}: {value}")

    print("\nPERFORMANCE")
    print("-" * 65)

    processing_ms = result.timing[
        "pipeline_processing_ms"
    ]

    print(
        f"Pipeline processing: "
        f"{processing_ms / 1000:.3f} seconds"
    )

    print("\nGENERATED ARTIFACTS")
    print("-" * 65)

    for name, artifact in result.artifacts.items():

        print(
            f"{artifact.filename:<38} "
            f"{artifact.size_bytes / 1024:>10.2f} KiB"
        )

    if result.warnings:

        print("\nWARNINGS")
        print("-" * 65)

        for warning in result.warnings:
            print(f"- {warning}")

    print("\nOutput directory:")
    print(result.output_dir)

    print("\nLIMITATIONS")
    print("-" * 65)

    print("Predicted water is not confirmed flood extent.")
    print("Results are intended for research use.")

    print("=" * 65)


def print_error(args, error, exit_code):

    if args.json:

        payload = {
            "status": "failed",
            "error": {
                "type": type(error).__name__,
                "message": str(error)
            },
            "exit_code": exit_code
        }

        print(
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False
            ),
            file=sys.stderr
        )

    else:

        print(
            f"ERROR [{type(error).__name__}]: {error}",
            file=sys.stderr
        )


def main(argv=None):

    parser = build_parser()

    args = parser.parse_args(argv)

    try:

        config = build_config(args)

        output_dir = resolve_output_dir(args)

        pipeline = FloodInferencePipeline(
            bundle_dir=args.bundle_dir,
            config=config
        )

        export_rasters = (
            not args.no_rasters
            and not args.inference_only
        )

        export_vectors = (
            not args.no_vectors
            and not args.inference_only
        )

        calculate_area = (
            not args.no_area
            and not args.inference_only
        )

        result = pipeline.predict(
            input_path=args.input,
            output_dir=output_dir,
            export_rasters=export_rasters,
            export_vectors=export_vectors,
            calculate_area=calculate_area
        )

        if args.json:

            print(result.to_json())

        else:

            print_summary(result)

        return 0

    except (InferenceError, ValueError) as exc:

        if args.debug:
            raise

        print_error(args, exc, 1)

        return 1

    except KeyboardInterrupt:

        print(
            "Analysis interrupted.",
            file=sys.stderr
        )

        return 130

    except Exception as exc:

        if args.debug:
            raise

        print_error(args, exc, 3)

        return 3


if __name__ == "__main__":
    raise SystemExit(main())