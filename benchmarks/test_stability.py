import json
import argparse


def main(args):
    with open(args.file, 'r') as f:
        data = json.load(f)
    tpt = data.get('request_throughput')
    assert tpt is not None
    result = "STABLE" if (tpt/args.rate >= 0.85) else "UNSTABLE"
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test completion time data for stability'
    )
    parser.add_argument(
        '--rate',
        type=float,
        default=1.0,
        help='Rate of arrival.(reqs/second)'
    )
    parser.add_argument(
        '--file',
        type=str,
        default=None,
        help='Input data file(in json format)'
    )
    args = parser.parse_args()
    main(args)
