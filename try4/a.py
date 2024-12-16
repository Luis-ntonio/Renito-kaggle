from gemma import *

scorer = PerplexityCalculator(model_path="../model/gemma", load_in_8bit=False)

text = ['reindeer', 'mistletoe', 'gingerbread', 'chimney', 'elf', 'ornament', 'fireplace', 'advent', 'scrooge', 'family']
text = ' '.join(text)

print(scorer.get_perplexity(text))