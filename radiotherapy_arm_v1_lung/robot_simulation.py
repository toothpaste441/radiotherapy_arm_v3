"""Command-line entry point for the CyberKnife tracking project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.live_demo import launch_live_demo
from src.simulation import (
    EXPERIMENTS,
    _write_plot,
    _write_summary,
    run_experiment,
    run_experiments,
)


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a six-DoF arm tracking a breathing-driven target."
    )
    parser.add_argument(
        "--experiment",
        choices=("all", *EXPERIMENTS.keys()),
        default="all",
        help="Experiment to run (default: all five headless experiments).",
    )
    parser.add_argument(
        "--view",
        action="store_true",
        help="Open the continuous viewer and interactive joint/trajectory controls.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for CSV metrics and plots (default: ./results).",
    )
    parser.add_argument("--plan", action="store_true",
                        help="RRT-Connect + quintic approach followed by breathing tracking")
    parser.add_argument("--plan-view", action="store_true",
                        help="Display the two-stage planning demonstration in MuJoCo")
    parser.add_argument('--brain', action='store_true',
                        help='Run synthetic patient-table / brain tumour point-aiming experiment')
    parser.add_argument('--brain-view', action='store_true',
                        help='View patient-table / brain tumour point-aiming experiment in MuJoCo')
    parser.add_argument('--cyberknife', action='store_true',
                        help='Run multi-viewpoint CyberKnife-style brain-target aiming demo')
    parser.add_argument('--cyberknife-view', action='store_true',
                        help='View multi-viewpoint CyberKnife-style demo in MuJoCo')
    parser.add_argument('--cyberknife-record', action='store_true',
                        help='Record exactly one seven-node CyberKnife simulation to an MP4')
    parser.add_argument("--cyberknife-nodes", type=int, default=7,
                        help="Number of viewpoints, 1-13 (default: 7)")
    parser.add_argument("--cyberknife-zero-g", action="store_true",
                        help="Run gravity-free controller diagnostic; not a physically realistic arm")
    parser.add_argument('--lung', action='store_true',
                        help='Run synthetic observed-breathing lung targeting + static baseline')
    parser.add_argument('--lung-view', action='store_true',
                        help='View compensated synthetic lung tumour targeting in MuJoCo')
    parser.add_argument('--lung-record', action='store_true',
                        help='Record compensated lung targeting to MP4')
    parser.add_argument('--lung-vision',action='store_true',
                        help='Run synthetic fixed-camera eye-to-hand IBVS lung tracking')
    parser.add_argument('--lung-vision-view',action='store_true',
                        help='View IBVS respiratory compensation in MuJoCo')
    parser.add_argument('--lung-vision-record',action='store_true',
                        help='Record fixed-camera IBVS lung compensation to MP4')
    parser.add_argument('--lung-duration' , type=float, default=12.0,
                        help='Breathing compensation duration in seconds (default 12)')
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()

    if args.lung or args.lung_view or args.lung_record or args.lung_vision or args.lung_vision_view or args.lung_vision_record:
        if any((args.cyberknife,args.cyberknife_view,args.cyberknife_record,
                args.brain,args.brain_view,args.plan,args.plan_view,args.view)):
            parser.error('--lung mode is separate from other simulation modes')
        from src.lung_tracking_demo import run_lung_tracking
        result = run_lung_tracking(PROJECT_ROOT,view=args.lung_view or args.lung_vision_view,
                                   record=args.lung_record or args.lung_vision_record,
                                   vision_mode=bool(args.lung_vision or args.lung_vision_view or args.lung_vision_record),
                                   duration_s=args.lung_duration,
                                   compare=not (args.lung_view or args.lung_vision_view))
        if not result['compensated']['successful']:
            raise SystemExit(1)
        return

    if args.cyberknife or args.cyberknife_view or args.cyberknife_zero_g or args.cyberknife_record:
        if args.brain or args.brain_view or args.plan or args.plan_view or args.view or args.experiment != 'all':
            parser.error('--cyberknife is a separate simulation mode')
        from src.cyberknife_motion_demo import run_cyberknife_demo
        result = run_cyberknife_demo(PROJECT_ROOT, view=args.cyberknife_view,
                                    nodes=args.cyberknife_nodes,
                                    zero_gravity=args.cyberknife_zero_g,
                                    record=args.cyberknife_record)
        if not result["successful"]:
            raise SystemExit(1)
        return

    if args.brain or args.brain_view:
        if args.plan or args.plan_view or args.view or args.experiment != 'all':
            parser.error('--brain is separate from the other simulation modes')
        from src.brain_demo import run_brain_demo
        run_brain_demo(PROJECT_ROOT, view=args.brain_view)
        return

    if args.plan or args.plan_view:
        if args.experiment != "all" or args.view:
            parser.error("--plan is separate from --experiment and --view")
        from src.planning_experiments import run_planning_demo
        run_planning_demo(PROJECT_ROOT, view=args.plan_view)
        return

    if args.view:
        if args.experiment != "all":
            parser.error("--experiment applies to headless runs; use the live buttons instead.")
        launch_live_demo(PROJECT_ROOT / "cyberknife_6dof.xml")
        print("Live simulation closed.")
        return
    elif args.experiment == "all":
        summaries = run_experiments(PROJECT_ROOT, output_dir)
    else:
        summary = run_experiment(
            PROJECT_ROOT / "cyberknife_6dof.xml",
            output_dir,
            EXPERIMENTS[args.experiment],
        )
        _write_summary(output_dir, [summary])
        _write_plot(output_dir, [summary])
        summaries = [summary]

    print(json.dumps(summaries, indent=2))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
