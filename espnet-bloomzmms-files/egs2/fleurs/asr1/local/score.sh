#!/usr/bin/env bash
score_lang_id=true
score_bleu=false
decode_folder=
. utils/parse_options.sh

if [ $# -gt 1 ]; then
    echo "Usage: $0 --score_lang_id ture [exp]" 1>&2
    echo ""
    echo "Language Identification Scoring."
    echo 'The default of <exp> is "exp/".'
    exit 1
fi

[ -f ./path.sh ] && . ./path.sh
set -euo pipefail
if [ $# -eq 1 ]; then
    exp=$1
else
    exp=exp
fi

if [ "${score_lang_id}" = true ]; then
	python local/score_lang_id.py --exp_folder $exp
fi

if [ "${score_bleu}" = true ]; then
	for ref in $(find ${exp} -name score_wer); do
		hypdir=$(dirname ${ref})
		sacrebleu \
			<(cut -f 1 ${ref}/ref.trn) \
			-i <(cut -f2- -d' ' ${hypdir}/text) \
			-m bleu chrf --format text > ${hypdir}/score_bleu.txt
		perl -p -e '$_="'${hypdir}' $_";' ${hypdir}/score_bleu.txt
	done
fi
