from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Bidirectional, Attention, Concatenate
from tensorflow.keras.optimizers import Adam

def model(vocab_size, embedding_dim, hidden_units, maxlen_questions, maxlen_answers):
    # Encoder
    encoder_inputs = Input(shape=(maxlen_questions,))
    encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(encoder_inputs)
    encoder_bidirectional = Bidirectional(LSTM(hidden_units, return_sequences=True, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_bidirectional(encoder_embedding)
    encoder_states = [Concatenate()([forward_h, backward_h]), Concatenate()([forward_c, backward_c])]

    # Decoder
    decoder_inputs = Input(shape=(maxlen_answers,))
    decoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = LSTM(hidden_units*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)

    # Attention Mechanism
    attention = Attention()([decoder_outputs, encoder_outputs])

    # Combine decoder outputs and attention context vector
    decoder_concat = Concatenate(axis=-1)([decoder_outputs, attention])

    # Dense layer for generating output
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model