from transformers import pipeline
print("Download model...")
classifier = pipeline("text-classification",model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)
print("Download successfully")

sentence = input("Please input english sentence (input \'EXIT\' to leave)\n> ")
while sentence != 'EXIT':
  prediction = classifier(sentence, return_all_scores = False)
  detail = classifier(sentence)
  print(f"Text: {sentence}")
  print(f"Most likely emotion is {prediction[0]['label']}.")
  print("------- Chances of all emotions -------")
  for var in detail[0]:
    print(f"{var['label']} - {var['score']}")
  sentence = input("Please input english sentence (input \'EXIT\' to leave)\n> ")





