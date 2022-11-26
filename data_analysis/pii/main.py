"""Here we detect PII: Emails, IP addresses, and keys (SSH/API) and redact/anonymize them
    * we use one regex for emails and one for IP addresses
    * for keys we use detect-secrets tool, which is a combination of multiple plgins (regexes, entropy..)
    * we also add some filters on top of each tool to decrease the number of false positives
"""

import argparse
import random
import json
from pprint import pprint as pp

from datasets import load_dataset

from pii_detection import scan_pii_batch
from pii_redaction import redact_pii_batch, random_replacements


def parseArgs():
    parser = argparse.ArgumentParser(description="PII detection and redaction")
    parser.add_argument(
        "--dataset_name",
        default="bigcode/pii-for-code",
        type=str,
        help="HF repo name/path of the dataset.",
    )
    parser.add_argument(
        "--subset",
        default="data/",
        type=str,
        help="Data subset to use.",
    )
    parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="Dataset split to process",
    )
    parser.add_argument(
        "--batch_size",
        default=100,
        type=int,
        help="Batch size for the PII detection/redaction",
    )
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="Seed for random",
    )
    parser.add_argument(
        "--num_proc",
        default=96,
        type=int,
        help="Number of processes to use for the PII detection/redaction",
    )
    parser.add_argument(
        "--perform_redaction",
        action="store_false",
        help="Whether to perform redaction of PII",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the dataset to the Hub",
    )
    # add argument for nale of dataset on the hub
    parser.add_argument(
        "--target_dataset",
        default="PII_dataset",
        type=str,
        help="HF repo name of the target dataset.",
    )
    # add an option of evaluating the pipeline on the PII benchmark we built
    return parser.parse_args()


def main():
    args = parseArgs()
    ds = load_dataset(args.dataset_name, data_dir=args.subset, split=args.split, use_auth_token=True)

    # scan the dataset for PII
    print("Starting PII detection...")
    ds_pii = ds.map(
        scan_pii_batch, batched=True, batch_size=args.batch_size, num_proc=args.num_proc, load_from_cache_file=False
    )
    print(f"Dataset after PII detection:\n{ds_pii}")
    print(f"Number of samples that contained PII: {sum(ds_pii['has_secrets'])}")
    print(f"Total number of secrets found: {sum(ds_pii['number_secrets'])}")

    # redact PII in the dataset
    if args.perform_redaction:
        print("Starting PII redaction...")
        random.seed(args.seed)

        # we use random replacements by default
        replacements = random_replacements()
        pp(f"replacements:\n{replacements}")
        with open("pii_replacements.json", "w") as f:
            json.dump(replacements, f)
        
        ds_pii = ds_pii.map(
            lambda x: redact_pii_batch(x, replacements),
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_proc,
            load_from_cache_file=False
        )
        print(f"Dataset after PII redaction:\n{ds_pii}")

    if args.push_to_hub:
        ds_pii.push_to_hub(args.target_dataset)


if __name__ == "__main__":
    main()