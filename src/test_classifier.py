import sys, os


parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent + '/../MITIE/mitielib')

from mitie import *

if len(sys.argv) < 3:
	print "No text for classifying. Exiting.."
	sys.exit()

text = sys.argv[2]
classifier = sys.argv[1]

tokens = tokenize(text)
print tokens

fe_filename= "../MITIE/MITIE-models/english/total_word_feature_extractor.dat"
trainer = text_categorizer_trainer(fe_filename)

cat = text_categorizer(classifier, fe_filename)


# Call the categorizer with a list of tokens, the response is a label (a string)
# and a score (a number) indicating the confidence of the categorizer
label, score = cat(tokens)
print(label,score)

