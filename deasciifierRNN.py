import codecs
import dynet as dy
import numpy as np
import pickle
import sys
EOS = "<EOS>" #all strings will end with the End Of String token

import dynet_config
dynet_config.set(
     random_seed=42,
     weight_decay=0.000001
)

MAX_STRING_LEN = 400
RNN_BUILDER = dy.LSTMBuilder
RNN_NUM_OF_LAYERS = 2
EMBEDDINGS_SIZE = 12
STATE_SIZE = 24

def get_train_val_data():
    f_input = codecs.open('Data/data_input.txt', encoding='utf-8')
    text_input = f_input.read()

    f_output = codecs.open('Data/data_output.txt', encoding='utf-8')
    text_output = f_output.read()

    input_sentences = text_input.split("\n")
    output_sentences = text_output.split("\n")

    train_set = list(zip(input_sentences[0:630000], output_sentences[0:630000]))
    val_set = list(zip(input_sentences[630000:-1], output_sentences[630000:-1]))

    return train_set,val_set

def prepare_data():
    f_input = codecs.open('Data/total_input.txt', encoding='utf-8')
    text_input = f_input.read()

    f_output = codecs.open('Data/total_output.txt', encoding='utf-8')
    text_output = f_output.read()

    chars = list(set(text_input+"\n"+text_output))
    chars.append(EOS)
    VOCAB_SIZE = len(chars)
    int2char = {ix:char for ix, char in enumerate(chars)}
    char2int = {char:ix for ix, char in enumerate(chars)}

    return VOCAB_SIZE,int2char,char2int

def train(network, train_set, val_set, epochs=3):
    def get_val_set_loss(network, val_set):
        loss = [network.get_loss(input_string, output_string).value() for input_string, output_string in val_set]
        return sum(loss)

    train_set = train_set * epochs
    trainer = dy.SimpleSGDTrainer(network.model)
    losses = []
    iterations = []
    for i, training_example in enumerate(train_set):
        input_string, output_string = training_example

        loss = network.get_loss(input_string, output_string)
        loss_value = loss.value()
        loss.backward()
        trainer.update()

        if i%1000 == 0:
            print(str(i)+".th example")

        # Accumulate average losses over training to plot
        if i % (len(train_set) / 100) == 0:
            val_loss = get_val_set_loss(network, val_set)
            losses.append(val_loss)
            iterations.append(i / ((len(train_set) / 100)))

            print('loss on validation set:', val_loss)

class SimpleRNNNetwork:
    def __init__(self, rnn_num_of_layers, embeddings_size, state_size,vocab_size,char2int,int2char):
        self.model = dy.ParameterCollection()

        # the embedding paramaters
        self.embeddings = self.model.add_lookup_parameters((vocab_size, embeddings_size))

        # the rnn
        self.RNN = RNN_BUILDER(rnn_num_of_layers, embeddings_size, state_size, self.model)

        # project the rnn output to a vector of VOCAB_SIZE length
        self.output_w = self.model.add_parameters((vocab_size, state_size))
        self.output_b = self.model.add_parameters((vocab_size))
        self.int2char = int2char
        self.char2int = char2int

    def load_model(self,filename):
        self.model = dy.ParameterCollection()
        self.output_w, self.output_b, self.embeddings, self.RNN = dy.load(filename, self.model)

    def save_model(self,filename):
        dy.renew_cg()
        with open(filename+"_output_w.txt", "w") as f:
            f.write(np.array_str(dy.parameter(self.output_w).npvalue()) + "\n")
        with open(filename+"_output_b.txt", "w") as f:
            f.write(np.array_str(dy.parameter(self.output_b).npvalue()) + "\n")
        dy.save(
            "Models/"+filename+"_train.model",
            [self.output_w,self.output_b,self.embeddings,self.RNN]     
        )

    def _preprocess(self, string):
        string = list(string) + [EOS]
        return [self.char2int[c] for c in string]

    def _embed_string(self, string):
        return [self.embeddings[char] for char in string]

    def _run_rnn(self, init_state, input_vecs):
        s = init_state

        states = s.add_inputs(input_vecs)
        rnn_outputs = [s.output() for s in states]
        return rnn_outputs

    def _get_probs(self, rnn_output):
        output_w = dy.parameter(self.output_w)
        output_b = dy.parameter(self.output_b)

        probs = dy.softmax(output_w * rnn_output + output_b)
        return probs

    def get_loss(self, input_string, output_string):
        input_string = self._preprocess(input_string)
        output_string = self._preprocess(output_string)

        dy.renew_cg()

        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)
        loss = []
        for rnn_output, output_char in zip(rnn_outputs, output_string):
            probs = self._get_probs(rnn_output)
            loss.append(-dy.log(dy.pick(probs, output_char)))
        loss = dy.esum(loss)
        return loss

    def _predict(self, probs):
        probs = probs.value()
        predicted_char = self.int2char[probs.index(max(probs))]
        return predicted_char

    def generate(self, input_string):
        input_string = self._preprocess(input_string)

        dy.renew_cg()

        embedded_string = self._embed_string(input_string)
        rnn_state = self.RNN.initial_state()
        rnn_outputs = self._run_rnn(rnn_state, embedded_string)

        output_string = []
        for rnn_output in rnn_outputs:
            probs = self._get_probs(rnn_output)
            predicted_char = self._predict(probs)
            output_string.append(predicted_char)
        output_string = ''.join(output_string)
        return output_string.replace('<EOS>', '')

def prepare_test_data():
    f_input = codecs.open('Data/data_input_test.txt', encoding='utf-8')
    text_input = f_input.read()

    f_output = codecs.open('Data/data_output_test.txt', encoding='utf-8')
    text_output = f_output.read()

    input_sentences = text_input.split("\n")
    output_sentences = text_output.split("\n")

    test_set = list(zip(input_sentences, output_sentences))

    return test_set

def test(model):
    from nltk.tokenize import word_tokenize

    try:
        test_set =  prepare_test_data()
        total_word_count = 0
        predicted_true = 0
        predicted_false = 0

        for i, test_instance in enumerate(test_set):
            input_string, output_string = test_instance
            predicted_sentence = model.generate(input_string)
            predicted_words = word_tokenize(predicted_sentence, language="turkish")
            output_words = word_tokenize(output_string, language="turkish")

            if len(output_words) == len(predicted_words):

                print("Output string: "+ output_string)
                print("Predicted string: "+ predicted_sentence)
                print("\n")

                for index in range(0,len(predicted_words)):
                    total_word_count += 1
                    if predicted_words[index] == output_words[index]:
                        predicted_true += 1
                    else:
                        predicted_false += 1

        print("Total word count: "+str(total_word_count))
        print("True predicted word count: " + str(predicted_true))
        print("False predicted word count: " + str(predicted_false))
        print("Accuracy: " + str(predicted_true/total_word_count))

    except Exception as e:
        print(e)
        print(output_string)
        print(predicted_sentence)

def test_model():
    char2int = pickle.load(open("char2int.p", "rb"))
    int2char = pickle.load(open("int2char.p", "rb"))
    VOCAB_SIZE = pickle.load(open("vocab_size.p", "rb"))

    rnn = SimpleRNNNetwork(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE,VOCAB_SIZE,char2int,int2char)
    rnn.load_model("Models/after_train.model")
    test(rnn)

def train_model():
    VOCAB_SIZE,int2char,char2int = prepare_data()
    train_set,val_set = get_train_val_data()
    pickle.dump( char2int, open( "char2int.p", "wb" ) )
    pickle.dump( int2char, open( "int2char.p", "wb" ) )
    pickle.dump( VOCAB_SIZE, open( "vocab_size.p", "wb" ) )

    rnn = SimpleRNNNetwork(RNN_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE,VOCAB_SIZE,char2int,int2char)

    train(rnn, train_set, val_set)
    rnn.save_model("after")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please run python file with one option : \"train\" or \"test\"")
    else:
        if str(sys.argv[1]) == "train":
            print('Training: ')
            train_model()
        elif str(sys.argv[1]) == "test":
            print('Test: ')
            test_model()
        else:
            print("Wrong parameter. Please run python file with the option \"train\" or \"test\"")
