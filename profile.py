# First generate profile_result.txt file by running the code in profiler mode
# python -m cProfile -o profile_results.txt run.py train --train-src=./en_es_data/train_tiny.es --train-tgt=./en_es_data/train_tiny.en \
#        --dev-src=./en_es_data/dev_tiny.es --dev-tgt=./en_es_data/dev_tiny.en --vocab=vocab_tiny_q2.json --batch-size=2 \
#        --max-epoch=201 --valid-niter=100
# And then run: python profile.py
import pstats
p = pstats.Stats('profile_results.txt')
#p.sort_stats('cumulative').print_stats(50)
p.sort_stats('tottime').print_stats(50)
