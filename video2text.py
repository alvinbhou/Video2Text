import numpy as np
import os, sys
import pickle, functools, operator
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Permute, Reshape, Activation, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse, json

def parse():
    parser = argparse.ArgumentParser(description="Video to Text Model")
    parser.add_argument('--uid', type=str, help='training uid', required=True)
    parser.add_argument('--train_path', default='data/training_data', type=str, help='training data path')
    parser.add_argument('--test_path', default='data/testing_data', type=str, help='test data path')
    parser.add_argument('--learning_rate', type=float, default=0.0007, help='learning rate for training')
    parser.add_argument('--batch_size', type=int, default=320, help='batch size for training')
    parser.add_argument('--epoch', type=int, default=100, help='epochs for training')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

class Video2Text(object):
    ''' Initialize the parameters for the model '''
    def __init__(self, args):
        self.uid = args.uid
        self.train_path = args.train_path
        self.test_path = args.test_path
        self.batch_size = args.batch_size
        self.lr = args.learning_rate
        self.epochs = args.epoch
        self.latent_dim = 512
        self.num_encoder_tokens = 4096
        self.num_decoder_tokens = 1500
        self.time_steps_encoder = 80
        self.time_steps_decoder = None
        self.preload = True
        self.preload_data_path = 'preload_data'
        self.trainable = False
        self.max_propablity = -1

        # processed data
        self.encoder_input_data = []
        self.decoder_input_data = []
        self.decoder_target_data = []
        self.tokenizer = None

        # models
        self.encoder_model = None
        self.decoder_model = None
        self.inf_encoder_model = None
        self.inf_decoder_model = None
        self.save_model_path = os.path.join('models', self.uid)


    def load_data(self):
        if(self.preload):
            with open(os.path.join(self.preload_data_path, 'X_data1024.jlib'), 'rb') as file:
                self.encoder_input_data = joblib.load(file)
                print(self.encoder_input_data.shape)
            with open(os.path.join(self.preload_data_path, 'y_data1024.jlib'), 'rb') as file:
                decoder_data = joblib.load(file)
                print(decoder_data.shape)
            with open(os.path.join(self.preload_data_path, 'tokenizer1024'), 'rb') as file:
                self.tokenizer = joblib.load(file)
                print(len(self.tokenizer.word_index))
            for e in decoder_data:
                i = e[:-1]
                o = e[1:]
                self.decoder_input_data.append(i)
                self.decoder_target_data.append(o)
            self.decoder_input_data = np.array(self.decoder_input_data)
            self.decoder_target_data = np.array(self.decoder_target_data)
        else:
            TRAIN_LABEL_PATH = os.path.join(self.train_path, 'training_label.json')
            with open(TRAIN_LABEL_PATH) as data_file:    
                y_data = json.load(data_file)
            videoId = []
            videoSeq = []
            for y in y_data:
                for idx, cap in enumerate(y['caption']):
                    cap = "<bos> " + cap + " <eos>"
                    videoId.append(y['id'])
                    videoSeq.append(cap)
            TRAIN_FEATURE_DIR = os.path.join(self.train_path, 'feat')
            x_data = {}
            for filename in os.listdir(TRAIN_FEATURE_DIR):
                f = np.load(os.path.join(TRAIN_FEATURE_DIR, filename))
                x_data[filename[:-4]] = f
            self.tokenizer = Tokenizer(num_words=self.num_decoder_tokens)
            self.tokenizer.fit_on_texts(videoSeq)
            word_index = self.tokenizer.word_index   
            print ('Convert to index sequences.')
            train_sequences = self.tokenizer.texts_to_sequences(videoSeq)
            train_sequences = np.array(train_sequences)
            train_sequences = pad_sequences(train_sequences, padding='post',truncating='post')
            print(train_sequences.shape)
            max_seq_length = train_sequences.shape[1]
            filesize = len(train_sequences)
            X_data = []
            y_data = []
            vCount = 0
            curFilename = videoId[0]
            for idx in  range(0,filesize):
                if(videoId[idx] == curFilename):
                    vCount = vCount + 1
                    if(vCount > 2):
                        continue
                else:
                    vCount = 1
                    curFilename = videoId[idx]
                self.encoder_input_data.append(x_data[videoId[idx]])
                y = to_categorical(train_sequences[idx], self.num_decoder_tokens)
                self.decoder_input_data.append(y[:-1])
                self.decoder_target_data.append(y[1:])
            self.encoder_input_data = np.array(self.encoder_input_data)
            self.decoder_input_data = np.array(self.decoder_input_data)
            self.decoder_target_data = np.array(self.decoder_target_data)
        # init decoder max length
        self.time_steps_decoder = self.decoder_input_data.shape[1]

        return [self.encoder_input_data, self.decoder_input_data], self.decoder_target_data, self.tokenizer

    def load_inference_models(self):
        # inference encoder model
        self.inf_encoder_model = load_model(os.path.join(self.save_model_path, 'encoder_model.h5'))

        # inference decoder model
        decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.inf_decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        self.inf_decoder_model.load_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))

    def train(self):
        # Define an input sequence and process it.
        encoder_inputs = Input(shape=(self.time_steps_encoder, self.num_encoder_tokens), name="encoder_inputs")
        encoder = LSTM(self.latent_dim, return_state=True,return_sequences=True, name='endcoder_lstm')
        _, state_h, state_c = encoder(encoder_inputs)
        encoder_states = [state_h, state_c]

        # Attention mechanism
        # attention = keras.layers.Permute((2,1))(encoder_output)
        # attention = keras.layers.Dense(TIME_STEPS_ENCODER, activation='softmax')(attention)
        # attention = keras.layers.Permute((2,1))(attention)
        # hidden = keras.layers.Multiply()([encoder_output, attention])
        # hidden = keras.layers.Permute((2,1))(hidden)
        # hidden = keras.layers.Dense(DECODER_MAX_LENGTH, activation='relu')(hidden)
        # hidden = keras.layers.Permute((2,1))(hidden)
        # hidden = keras.layers.Dense(num_decoder_tokens, activation='relu')(hidden)

        # Set up the decoder
        decoder_inputs = Input(shape=(self.time_steps_decoder, self.num_decoder_tokens), name= "decoder_inputs")
        decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='relu', name='decoder_relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()

        # Early Stopping
        earlystopping = EarlyStopping(monitor='val_loss', patience = 4, verbose=1, mode='min')

        # Run training
        opt = keras.optimizers.adam(lr = self.lr)
        model.compile(metrics=['accuracy'], optimizer=opt, loss='categorical_crossentropy')
        try:
            model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_split=0.15,
                    callbacks=[earlystopping])
        except KeyboardInterrupt:
            print("\nW: interrupt received, stopping")
        finally:
            pass
    
        # saving process
        if not os.path.exists(self.save_model_path):
            os.makedirs(self.save_model_path)
        
        self.encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        self.encoder_model.summary()
        self.decoder_model.summary()

        # save models
        self.encoder_model.save(os.path.join(self.save_model_path, 'encoder_model.h5'))
        self.decoder_model.save_weights(os.path.join(self.save_model_path, 'decoder_model_weights.h5'))
        with open(os.path.join(self.save_model_path,'tokenizer'+ str(self.num_decoder_tokens) ),'wb') as file:
            joblib.dump(self.tokenizer, file)
        # attention_model.save(os.path.join(directory, 'attention_model.h5'))

    def decode_sequence2bs(self, input_seq):
        states_value = self.inf_encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.tokenizer.word_index['bos']] = 1
        self.beam_search(target_seq, states_value,[],[],0)
        return decode_seq

    def beam_search(self, target_seq, states_value, prob,  path, lens):
        global decode_seq
        node = 2
        output_tokens, h, c = self.inf_decoder_model.predict(
            [target_seq] + states_value)
        output_tokens = output_tokens.reshape((self.num_decoder_tokens))
        sampled_token_index = output_tokens.argsort()[-node:][::-1]
        states_value = [h, c]
        for i in range(node):
            if sampled_token_index[i] == 0:
                sampled_char = ''
            else:
                sampled_char = list(self.tokenizer.word_index.keys())[list(self.tokenizer.word_index.values()).index(sampled_token_index[i])]
            MAX_LEN = 9
            if(sampled_char != 'eos' and lens <= MAX_LEN):
                p = output_tokens[sampled_token_index[i]]
                if(sampled_char == ''):
                    p = 1
                prob_new = list(prob)
                prob_new.append(p)
                path_new = list(path)
                path_new.append(sampled_char)
                target_seq = np.zeros((1, 1, self.num_decoder_tokens))
                target_seq[0, 0, sampled_token_index[i]] = 1.
                self.beam_search(target_seq, states_value, prob_new, path_new, lens+1)
            else:
                p = output_tokens[sampled_token_index[i]]
                prob_new = list(prob)
                prob_new.append(p)
                p = functools.reduce(operator.mul, prob_new, 1)
                if(p > self.max_propablity):
                    decode_seq = path
                    self.max_propablity = p

    def decoded_sentence_tuning(self, decoded_sentence):
        decode_str = []
        filter_string = ['bos', 'eos']
        unigram = {}
        last_string = ""
        for idx2, c in enumerate(decoded_sentence):
            if c in unigram:
                unigram[c] += 1
            else:
                unigram[c] = 1
            if(last_string == c and idx2 > 0):
                continue
            if c in filter_string:
                continue
            if len(c) > 0:
                decode_str.append(c)
            if idx2 > 0:
                last_string = c
        return decode_str

    def get_test_data(self, path):
        X_test = []
        X_test_filename = []
        with open (os.path.join(path, 'testing_id.txt')) as testing_file:
            lines = testing_file.readlines()
            for filename in lines:
                filename = filename.strip()
                f = np.load(os.path.join(path , 'feat', filename + '.npy'))
                X_test.append(f)
                X_test_filename.append(filename[:-4])
            X_test = np.array(X_test)
        return X_test, X_test_filename

    def test(self):
        X_test, X_test_filename = self.get_test_data(os.path.join(self.test_path))
        # generate inference test outputs
        with open(os.path.join(self.save_model_path, 'test_output.txt'), 'w') as file:
            for idx, x in enumerate(X_test): 
                file.write(X_test_filename[idx]+',')
                decoded_sentence = self.decode_sequence2bs(x.reshape(-1, 80, 4096))
                decode_str = self.decoded_sentence_tuning(decoded_sentence)
                for d in decode_str:
                    file.write(d + ' ')
                file.write('\n')
                # re-init max prob
                self.max_propablity = -1
                
if __name__ == "__main__":
    vid2Text = Video2Text(parse())
    vid2Text.load_data()
    if(vid2Text.trainable):
        vid2Text.train()
    vid2Text.load_inference_models()
    vid2Text.test()
