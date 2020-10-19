import gzip
import pickle
import zlib
import argparse
import sys
import time
import math

MAX_LEVENSHTEIN_DISTANCE = 2
VITERBI_DICTIONARY_LIMIT = 2000


def parse_aol_data(clear_data_file_name):
    clear_data_file = open(clear_data_file_name, 'w', encoding='utf-8')
    prefix_name = './aol/AOL-user-ct-collection/user-ct-test-collection-'
    suffix_name = '.txt.gz'
    last_query = ''
    for number in range(10):
        name = prefix_name
        if number != 9:
            name += '0' + str(number + 1)
        else:
            name += '10'
        name += suffix_name

        first = True
        for line in gzip.open(name, mode='rt', encoding='utf-8'):
            if first:
                first = False
                continue

            tokens = line.split()
            query = ''
            if tokens[-1].startswith('http'):
                query = ' '.join(tokens[1:-4])
            else:
                query = ' '.join(tokens[1:-2])
            if query == last_query:
                continue

            result = []
            for character in query.lower():
                if character.isalpha() or character.isdigit():
                    result.append(character)
                else:
                    result.append(' ')

            clear_data_file.write(''.join(result) + '\n')
            last_query = query
    clear_data_file.close()


def learning(dataset_file_name):
    dataset_file = open(dataset_file_name, 'r', encoding='utf-8')
    unogramm_freq = dict()
    bigramm_freq = dict()

    for line in dataset_file:
        last_token = ''
        for token in line.lower().split():

            if token not in unogramm_freq.keys():
                unogramm_freq[token] = 1
            else:
                unogramm_freq[token] += 1
            if last_token != '':
                bigramm = last_token + ' ' + token
                if bigramm not in bigramm_freq.keys():
                    bigramm_freq[bigramm] = 1
                else:
                    bigramm_freq[bigramm] += 1
            last_token = token

    dataset_file.close()

    return unogramm_freq, bigramm_freq


def serialize_gramms(unogramm_freq, bigramm_freq, unogramm_freq_file_name, bigramm_freq_file_name):
    unogramm_freq_file = open(unogramm_freq_file_name, mode='wb')
    bigramm_freq_file = open(bigramm_freq_file_name, mode='wb')

    unogramm_freq_file.write(zlib.compress(pickle.dumps(unogramm_freq), level=9))
    bigramm_freq_file.write(zlib.compress(pickle.dumps(bigramm_freq), level=9))

    unogramm_freq_file.close()
    bigramm_freq_file.close()


def deserialize_gramms(unogramm_freq_file_name, bigramm_freq_file_name):
    unogramm_freq_file = open(unogramm_freq_file_name, mode='rb')
    bigramm_freq_file = open(bigramm_freq_file_name, mode='rb')

    unogramms_freq = pickle.loads(zlib.decompress(unogramm_freq_file.read()))
    bigramm_freq = pickle.loads(zlib.decompress(bigramm_freq_file.read()))

    unogramm_freq_file.close()
    bigramm_freq_file.close()

    return unogramms_freq, bigramm_freq


def p_conditional(word_2, word_1, unogramm_freq, bigramm_freq, lidstone_parameter=0.5):
    theoretical_bigramm_number = len(unogramm_freq) ** 2
    bigramm = word_1+' '+word_2
    if word_1 not in unogramm_freq.keys():
        return 0.0
    bigramm_number = 0
    if bigramm not in bigramm_freq.keys():
        bigramm_number = 0
    else:
        bigramm_number = bigramm_freq[bigramm]
    return (bigramm_number + lidstone_parameter) / (unogramm_freq[word_1] + theoretical_bigramm_number*lidstone_parameter)


def get_top_predictions(request, number_top, unogramm_freq, bigramm_freq, number_tokens):
    pure_request = []
    for character in request.lower():
        if character.isalpha() or character.isdigit():
            pure_request.append(character)
        else:
            pure_request.append(' ')

    terms = ''.join(pure_request).split()

    def wagner_fisher(a, b):
        N, M = len(a), len(b)
        D = []
        for i in range(M+1):
            D.append([0] * (N+1))
        D[0][0] = 0
        for i in range(1, N+1):
            D[0][i] = D[0][i-1] + 1
        for i in range(1, M+1):
            D[i][0] = D[i-1][0] + 1
            for j in range(1, N+1):
                D[i][j] = min(D[i-1][j]+1, D[i][j-1]+1, D[i-1][j-1]+(0 if b[i-1] == a[j-1] else 1))
        return D[M][N]

    word_levels = []
    for i in range(len(terms)):
        word_levels.append([])

    for word in list(unogramm_freq.keys()):
        for i in range(len(terms)):
            if len(word_levels[i]) > VITERBI_DICTIONARY_LIMIT or abs(len(word)-len(terms[i])) > MAX_LEVENSHTEIN_DISTANCE:
                continue
            if wagner_fisher(word, terms[i]) <= MAX_LEVENSHTEIN_DISTANCE:
                word_levels[i].append(word)
                break

    def my_viterbi_adaptation():
        T = len(terms)
        delta = []
        psi = []
        for i in range(T):
            delta.append([0.0]*len(word_levels[i]))
            psi.append([0.0] * len(word_levels[i]))
        for i in range(len(word_levels[0])):
            delta[0][i] = 0.0
            psi[0][i] = None

        for t in range(1, T):
            for j in range(len(word_levels[t])):
                max_value = delta[t - 1][0] + math.log2(p_conditional(word_levels[t][j], word_levels[t - 1][0], unogramm_freq,
                                                            bigramm_freq))
                max_index = 0
                for i in range(1, len(word_levels[t-1])):
                    value = delta[t - 1][i] + math.log2(p_conditional(word_levels[t][j], word_levels[t - 1][i], unogramm_freq,
                                                            bigramm_freq))
                    if max_value < value:
                        max_value = value
                        max_index = i
                delta[t][j] = max_value
                psi[t][j] = max_index

        end_value = delta[T-1][0]
        end_index = 0
        for i in range(1, len(word_levels[T-1])):
            if end_value < delta[T-1][i]:
                end_value = delta[T - 1][i]
                end_index = i

        path = []
        index = end_index
        for i in range(T-1, -1, -1):
            path.append(word_levels[i][index])
            index = psi[i][index]
        path.reverse()

        return [' '.join(path)]

    if len(terms) > 0:
        return my_viterbi_adaptation()
    return []


def to_format(number, precision=2):
    return format(number, '.'+str(precision)+'f')


def run():
    dataset_file_name = 'aol/aol_query.txt'
    # dataset_file_name = 'aol_query_test.txt'
    unogramm_freq_file_name = 'unogramm_freq.bin'
    bigramm_freq_file_name = 'bigramm_freq.bin'
    # parse_aol_data(dataset_file_name)
    # unogramm_freq, bigramm_freq = learning(dataset_file_name)
    # serialize_gramms(unogramm_freq, bigramm_freq, unogramm_freq_file_name, bigramm_freq_file_name)
    unogramm_freq, bigramm_freq = deserialize_gramms(unogramm_freq_file_name, bigramm_freq_file_name)
    print('Dictionary power:', len(unogramm_freq), 'words.')
    print('Bigramm number:', len(bigramm_freq), 'words.')
    number_tokens = 0
    for key in unogramm_freq.keys():
        number_tokens += unogramm_freq[key]

    com_args_parser = argparse.ArgumentParser()
    com_args_parser.add_argument('--interactive-mode', '-i', action="store_true", help='prediction by input requests')
    com_args_parser.add_argument('--file-mode', '-f', type=str, help='prediction by requests in file')
    com_args_parser.add_argument('--output-file', '-o', type=str, help='output file')
    com_args_parser.add_argument('--number-top', '-n', type=int, help='number top predictions')
    my_namespace = com_args_parser.parse_args()

    if my_namespace.number_top < 1:
        print('Negative number top!')
        exit(1)

    if my_namespace.interactive_mode:
        print('Request: ', end='', flush=True)
        for request in sys.stdin:
            start_time = time.time()
            predictions = get_top_predictions(request, my_namespace.number_top, unogramm_freq, bigramm_freq, number_tokens)
            request_time = time.time() - start_time
            print('\n============= Prediction time is ' + to_format(request_time, 4) + 's =============\n', flush=True)
            if predictions is None:
                print('Not exist predictions.', flush=True)
            else:
                print('Predictions:', flush=True)
                index = 1
                for prediction in predictions:
                    print(str(index)+'.', prediction, flush=True)
                    index += 1
            print('\nRequest: ', end='', flush=True)

if __name__ == "__main__":
    run()
