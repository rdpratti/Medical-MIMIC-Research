# Medical-MIMIC-Research
 
 This project supports my work exploring the MIT MIMIC database.

 I am trying to answer the question on whether a model can be built to help support diagnostic decisions within the first few days of admittance to an ICU.

The project has four main notebooks:

1) Mimic-EDA-Notebook			&ensp;&ensp;Explores details about admission and note data by diagnosis

2) Scrub-MIMIC-Data-2			&ensp;&ensp;Cleanup text data

3) Setup-For-Model-2		 	&ensp;&ensp;Creates, assigns labels, splits and balances files

4) Diagnosis-Ensemble			&ensp;&ensp;Runs Code Scripts that Fine-Tunes ClinicalBert, Test and Measure Performance

The first three can easily run in a CPU environment. The last one (4) was ported and executed on Google-Collab

There are 2 python scripts executed by the last notebook:

1) BERT-Diagnosis.py

2) TFIDF-Diagnosis.py


There is two additional File

1) Using Early Diagnosis Prediction in the ICU as a Trigger Data Point			&ensp;&ensp;Research Overview Presentation as a PDF

2) Abstract.pdf										&ensp;&ensp;Research Abstract
	


