
import sys, os, csv

parent = os.path.dirname(os.path.realpath(__file__))
sys.path.append(parent + '/../MITIE/mitielib')

from mitie import *

fe_filename= "../MITIE/MITIE-models/english/total_word_feature_extractor.dat"
trainer = text_categorizer_trainer(fe_filename)

if len(sys.argv) < 2:
	print "No training file specified. Exiting.."
	sys.exit()
else:
	rawDataPath = sys.argv[1]
	print "Opening " + rawDataPath + " ..."
	if os.path.isfile(rawDataPath) == False:
		print "Path specified is not valid. Exiting.."
		sys.exit()
	else:
		with open(rawDataPath, 'rb') as csvfile:
			csvReader = csv.reader(csvfile)
			for row in csvReader:
				tokens = tokenize(row[0])
				trainer.add_labeled_text(tokens, row[1])



trainer.num_threads = 4

cat = trainer.train()

#get filename
saveName = os.path.basename(rawDataPath)
#remove extension
saveName = saveName.split(".")[0]
#save
cat.save_to_disk("../classifiers/" + saveName + ".dat",pure_model=True)