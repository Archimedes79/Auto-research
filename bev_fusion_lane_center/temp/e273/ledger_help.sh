#!/bin/bash
cd "$(dirname "$0")/../.."
python3 ../autoresearch/tools/ledger.py note --help
python3 ../autoresearch/tools/ledger.py tail 3 | cut -c1-600
