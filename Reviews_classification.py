import sys
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow_datasets as tfds

sys.stdout.reconfigure(encoding='utf-8')

dataset, info = tfds.load('imdb_reviews', as_supervised=True, with_info=True)
train_data, test_data = dataset['train'], dataset['test']

train_sentences = []
train_labels = []
test_sentences = []
test_labels = []

for sentence, label in train_data:
    train_sentences.append(sentence.numpy().decode('utf8'))
    train_labels.append(label.numpy())


for sentence, label in test_data:
    test_sentences.append(sentence.numpy().decode('utf8'))
    test_labels.append(label.numpy())


vocab_size = 10000
max_length = 120
trunc_type = 'post'
oov_token = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_sentences)


train_sequences = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_sequences, maxlen=max_length, truncating=trunc_type)

test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length)


train_labels = tf.convert_to_tensor(train_labels)
test_labels = tf.convert_to_tensor(test_labels)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=16),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(
    train_padded, 
    train_labels, 
    epochs=5, 
    validation_data=(test_padded, test_labels), 
    verbose=1
)

loss, accuracy = model.evaluate(test_padded, test_labels)
print(f"Test Accuracy: {accuracy:.2f}")

sample_review = ["The movie was bad"]
sample_sequence = tokenizer.texts_to_sequences(sample_review)
sample_padded = pad_sequences(sample_sequence, maxlen=max_length, padding='post')

prediction = model.predict(sample_padded)
print(f"Sentiment: {'Positive' if prediction >= 0.5 else 'Negative'}")
