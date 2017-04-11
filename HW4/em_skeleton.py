# coding: utf-8
import numpy as np

"""
Outline of all probability and count variables
--------------------------------------------------
t: real counts for tag t
tw: real counts for word w given we know it has tag t
tpw: real counts for word w directly preceding a word with tag t
tnw: real counts for word w directly following a word with tag t

et: expected counts for tag t
etw: expected counts for word w given we know it has tag t
etpw: expected counts for word w directly preceding a word with tag t
etnw: expected counts for word w directly following a word with tag t

pt: p(t), probability of tag t (based on weighted combination of real and expected counts)
ptw: p_{0}(w_i|t_i)
ptpw: p_{-1}(w_{i-1}|t_i)
ptnw: p_{+1}(w_{i+1}|t_i)
"""


def load_train_data():
    """
    Loads the training corpus
    Output:
        word_list : a list of 'sentences', each element contains a list of indices corresponding to words
        label_list : a list of labeled 'sentences', each element contains a list of indices corresponding to tags
        vocmap : vocabulary:index map
        tagmap : tag:index map
    """
    # load the training corpus
    # load vocabulary and assign index
    vocmap = dict()
    with open('freqwords', 'r') as f:
        c = 0
        for line in f:
            vocmap[line.rstrip()] = c
            c += 1
        vocmap['UNKA'] = c
    print('\nVocabulary size: {}'.format(len(vocmap)))

    # load sentence and label
    with open('wsj.0-18', 'r') as f:
        word_list = []
        label_list = []
        for line in f:
            data = line.strip().split()
            tmp = data[0::2]
            word_list.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
            tmp = data[1::2]
            label_list.append(tmp)

    # construct tagset and assign index
    tagmap = dict()
    for sent in label_list:
        for i in range(len(sent)):
            if not sent[i] in tagmap:
                tagmap[sent[i]] = len(tagmap)
            sent[i] = tagmap[sent[i]]
    print('Tagset size: {}'.format(len(tagmap)))

    return word_list, label_list, vocmap, tagmap


def smoothing(alpha, t, tw, tpw, tnw):
    """
    Adding the smooth counts to the original ones
    """
    T, W = tw.shape
    t = t + alpha / T
    tw = tw + alpha / (T * W)
    tpw = tpw + alpha / (T * W)
    tnw = tnw + alpha / (T * W)
    return t, tw, tpw, tnw


def split_data_and_get_counts(ratio, word_list, label_list, vocmap, tagmap):
    """
    Split the fully labeled data by the ratio and return the count and all sentences
    Input:
        ratio : the split ratio of labeled and unlabled data; used to compute weight of real counts
        word_list : a list of 'sentences', each element contains a list of indices corresponding to words
        label_list : a list of labeled 'sentences', each element contains a list of indices corresponding to tags
        vocmap : vocabulary:index map
        tagmap : tag:index map

    Output:
        t: (T,1) array, real counts for tag t
        tw: (T,W) matrix, real counts for word w given we know it has tag t
        tpw: (T,W) matrix, real counts for word w directly preceding a word with tag t
        tnw: (T,W) matrix, real counts for word w directly following a word with tag t
        unlabel_word_list: a list of 'unlabeled' sentences, each element contains a list of words
        unlabel_label_list: Corresponding list of 'unlabeled' sentences, each element contains a list of indices corresponding to tags
    """
    label_num = int(ratio * len(word_list))
    unlabel_word_list = word_list[label_num:]
    unlabel_label_list = label_list[label_num:]
    word_list = word_list[:label_num]
    label_list = label_list[:label_num]

    # construct parameter table
    W = len(vocmap)
    T = len(tagmap)

    t = np.zeros(T)
    tw = np.zeros((T, W))
    tpw = np.zeros((T, W))
    tnw = np.zeros((T, W))

    # calculate count to the table
    for i in range(len(word_list)):
        word = word_list[i]
        label = label_list[i]
        for j in range(len(word)):
            t[label[j]] += 1.0
            tw[label[j], word[j]] += 1.0
            if j > 0:
                tpw[label[j], word[j - 1]] += 1.0
            if j < len(word) - 1:
                tnw[label[j], word[j + 1]] += 1.0

    # smoothing
    t, tw, tpw, tnw = smoothing(1.0, t, tw, tpw, tnw)

    return t, tw, tpw, tnw, unlabel_word_list, unlabel_label_list


def load_test_data(vocmap, tagmap):
    """
    Load and return the test data and gold truth labels, converted into indices
    Input:
        vocmap : vocabulary:index map
        tagmap : tag:index map
    Output:
        word_list : a list of 'sentences', each element contains a list of indices corresponding to words
        label_list : a list of labeled 'sentences', each element contains a list of indices corresponding to tags
    """
    # a list of sentences, each element contains a list of words
    word_list = []
    # a list of sentences, each element contains a list of labels (POS)
    label_list = []

    with open('wsj.19-21', 'r') as f:
        for line in f:
            data = line.strip().split()
            tmp = data[0::2]
            word_list.append([vocmap[s] if s in vocmap else vocmap['UNKA'] for s in tmp])
            tmp = data[1::2]
            label_list.append([tagmap[s] for s in tmp])

    return word_list, label_list


def M_step(et, etw, etpw, etnw, t, tw, tpw, tnw):
    """
    Input:
        Expected counts et, etw, etpw, etnw
        Real counts t, tw, tpw, tnw
    Output:
        Model parameters pt, ptw, ptpw, ptnw
    """
    T, W = etw.shape
    pt = np.zeros(et.shape)
    ptw = np.zeros(etw.shape)
    ptpw = np.zeros(etpw.shape)
    ptnw = np.zeros(etnw.shape)
    # print(t)
    # print(tw)
    # print(tpw)
    # print(tnw)

    # c is the weight of real count
    c = 100.0

    # Estimate parameters pt, ptw, ptpw, ptnw based on the expected counts and real counts

    # TODO: implement below
    #
    #
    #
    nt = c*t + et
    ntw = c*tw + etw
    ntpw = c*tpw + etpw
    ntnw = c*tnw + etnw
    for i in range(T):
        pt[i] = nt[i] / np.sum(nt)
    for i in range(T):
        for j in range(W):
            ptw[i][j] = ntw[i][j] / nt[i]
    for i in range(T):
        for j in range(W):
            ptpw[i][j] = ntpw[i][j] / nt[i]
    for i in range(T):
        for j in range(W):
            ptnw[i][j] = ntnw[i][j] / nt[i]


    return pt, ptw, ptpw, ptnw


def E_step_A(pt, ptw, ptpw, ptnw, word_list):
    """
    Input:
        Model parameters pt, ptw, ptpw, ptnw,
        word_list: : a list of 'sentences', each element contains a list of indices corresponding to words
    Output:
        Expected counts et, etw, etpw, etnw
    """
    T, W = ptw.shape
    # etpw is not computed in this function
    # etnw is not computed in this function
    et = np.zeros(T)
    etw = np.zeros((T, W))
    etpw = np.zeros((T, W))
    etnw = np.zeros((T, W))

    for sent in word_list:
        for pos in range(len(sent)):
            # Compute the posterior for each word and normalize
            # Accumulate expected counts based on posterior
            post = pt * ptw[:, sent[pos]] / np.dot(pt, ptw[:, sent[pos]])
            et += post
            etw[:, sent[pos]] += post
    return et, etw, etpw, etnw


def likelihood_A(pt, ptw, ptpw, ptnw, word_list, t, tw, tpw, tnw):
    """
    Compute likelihood based on Model A
    Input:
       word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
       t, tw, tpw, tnw are real counts
    Output:
        log-likelihood l
    """
    l = sum([sum([np.log(sum(pt * ptw[:, word])) for word in sent]) for sent in word_list])
    # log-prior likelihood, resulting in smoothing
    c = 100.0
    l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)))
    return l


def E_step_B(pt, ptw, ptpw, ptnw, word_list):
    """
    Input:
        Model parameters pt, ptw, ptpw, ptnw,
        word_list: : a list of 'sentences', each element contains a list of indices corresponding to words
    Output:
        Expected counts et, etw, etpw, etnw
    """
    T, W = ptw.shape

    # etnw is not computed in this function
    et = np.zeros(T)
    etw = np.zeros((T, W))
    etpw = np.zeros((T, W))
    etnw = np.zeros((T, W))

    # TODO: filled-in below
    for sent in word_list:
        for pos in range(len(sent)):
            # Compute the posterior for the first word or other words
            # Hint: the posterior formula for the first word and others are different
            # consider edge case first
            if pos == 0:
                # Compute the posterior for each word and normalize
                # Accumulate expected counts based on posterior
                post = pt * ptw[:, sent[pos]] / np.dot(pt, ptw[:, sent[pos]])
                et += post
                etw[:, sent[pos]] += post
            # consider non−edge case
            else:
                # Compute the posterior for each word and normalize
                post = pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]] / \
                    np.dot(pt, (ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]]))
                et += post
                etw[:, sent[pos]] += post
                etpw[:, sent[pos - 1]] += post

    return et, etw, etpw, etnw


def likelihood_B(pt, ptw, ptpw, ptnw, word_list, t, tw, tpw, tnw):
    """
    Compute likelihood based on Model B
    Input:
       word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
       t, tw, tpw, tnw are real counts
    Output:
        log-likelihood l
    """
    sum_x = 0.0
    sum_y = 0.0
    for sent in word_list:
        sum_x += sum_y
        sum_y = 0.0
        for i in range(len(sent)):
            if i > 0:
                sum_y += np.log(sum(pt * ptw[:, sent[i]] * ptpw[:, sent[i - 1]]))
            else:
                sum_y += np.log(sum(pt * ptw[:, sent[i]]))
    l = sum_x
    # log-prior likelihood, resulting in smoothing
    c = 100.0
    l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)))
    return l


def E_step_C(pt, ptw, ptpw, ptnw, word_list):
    """
    Input:
        Model parameters pt, ptw, ptpw, ptnw,
        word_list: : a list of 'sentences', each element contains a list of indices corresponding to words

    Output:
        Expected counts et, etw, etpw, etnw
    """
    T, W = ptw.shape

    et = np.zeros(T)
    etw = np.zeros((T, W))
    etpw = np.zeros((T, W))
    etnw = np.zeros((T, W))

    # TODO: filled-in below
    for sent in word_list:
        for pos in range(len(sent)):
            # Compute the posterior for the first word, middle word or last owrd
            # Hint: the posterior formula for the first word, the last word and others are different
            # consider edge case first
            if len(sent) == 1:
                post = pt * ptw[:, sent[pos]] / np.dot(pt, ptw[:, sent[pos]])
                etw[:, sent[pos]] += post
            elif pos == 0:
                post_1 = pt * ptw[:, sent[pos]] * ptnw[:, sent[pos + 1]]
                post_2 = np.dot(pt, (ptw[:, sent[pos]] * ptnw[:, sent[pos + 1]]))
                post = post_1 / post_2
                et += post
                etw[:, sent[pos]] += post
                etnw[:, sent[pos + 1]] += post
            elif pos == len(sent) - 1:
                post_1 = pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]]
                post_2 = np.dot(pt, (ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]]))
                post = post_1 / post_2
                et += post
                etw[:, sent[pos]] += post
                etpw[:, sent[pos - 1]] += post
            # consider non−edge case
            else:
                post_1 = pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]] * ptnw[:, sent[pos + 1]]
                post_2 = np.dot(pt, (ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]] * ptnw[:, sent[pos + 1]]))
                post = post_1 / post_2
                et += post
                etw[:, sent[pos]] += post
                etnw[:, sent[pos + 1]] += post
                etpw[:, sent[pos - 1]] += post

    return et, etw, etpw, etnw


def likelihood_C(pt, ptw, ptpw, ptnw, word_list, t, tw, tpw, tnw):
    """
    Compute likelihood based on Model C
    Input:
       word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
       t, tw, tpw, tnw are real counts
    Output:
        log-likelihood l
    """
    l = 0.0
    for sent in word_list:
        for pos in range(len(sent)):
            prob = pt * ptw[:, sent[pos]]
            if pos > 0:
                prob = prob * ptpw[:, sent[pos - 1]]
            if pos < len(sent) - 1:
                prob = prob * ptnw[:, sent[pos + 1]]
            l += np.log(sum(prob))
    # log-prior likelihood, resulting in smoothing
    c = 100.0
    l += c * (np.sum(t * np.log(pt)) + np.sum(tw * np.log(ptw)) + np.sum(tpw * np.log(ptpw)) + np.sum(tnw * np.log(ptnw)))

    return l


def predict_A(word_list, pt, ptw, ptpw, ptnw):
    """
    word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
    pred is the list of prediction, each element is a list of tag index predictions for each word in the sentence
    e.g. pred = [[1,2], [2,3]] for 2 sentences each with 2 words
    ptpw and ptnw are not used in this function
    """
    pred = []

    # Predict tag index in each sentence based on Model A
    for sent in word_list:
        cur_pred = []
        for pos in range(len(sent)):
            # pred_tag is the prediction of tag for the current word
            # Your code here:
            pred_tag = np.argmax(pt * ptw[:, sent[pos]])
            cur_pred.append(pred_tag)
        pred.append(cur_pred)
    return pred


def predict_B(word_list, pt, ptw, ptpw, ptnw):
    """
    word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
    pred is the list of predictions, each element is a list of tag index predictions for each word in the sentence
    e.g. pred = [[1,2], [2,3]] for 2 sentences each with 2 words
    ptnw is not used in this function
    """
    pred = []

    # Predict tag index in each sentence based on Model B
    for sent in word_list:
        cur_pred = []
        for pos in range(len(sent)):
            # pred_tag is the prediction of tag for the current word
            pred_tag = -1

            # Your code here:
            # Hint: note that the probability definition is different for the
            # first word and the rest
            if pos == 0:
                pred_tag = np.argmax(pt * ptw[:, sent[pos]])
            else:
                pred_tag = np.argmax(
                    pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]])
            # append the prediction to the list
            cur_pred.append(pred_tag)
        pred.append(cur_pred)

    return pred


def predict_C(word_list, pt, ptw, ptpw, ptnw):
    """
    word_list is the list for testing sentence; pt, ptw, ptpw, ptnw are parameters
    pred is the list of predictions, each element is a list of tag index predictions for each word in the sentence
    e.g. pred = [[1,2], [2,3]] for 2 sentences each with 2 words
    """
    pred = []

    # Predict tag index in each sentence based on Model C
    for sent in word_list:
        cur_pred = []
        pos = 0
        for pos in range(len(sent)):
            # pred_tag is the prediction of tag for the current word
            pred_tag = -1

            # Your code here:
            # Hint: note that the probability definition is different for the
            # first word, the last word and the middle words
            if len(sent) == 1:
                pred_tag = np.argmax(pt * ptw[:, sent[pos]])
            elif pos == 0:
                pred_tag = np.argmax(
                    pt * ptw[:, sent[pos]] * ptnw[:, sent[pos + 1]])
            elif pos == len(sent) - 1:
                pred_tag = np.argmax(
                    pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]])
            else:
                pred_tag = np.argmax(pt * ptw[:, sent[pos]] * ptpw[:, sent[pos - 1]] * ptnw[:, sent[pos + 1]])

            # append the prediction to the list
            cur_pred.append(pred_tag)
        pred.append(cur_pred)

    return pred


def evaluate(label_list, pred):
    """
    Compute accuracy
    Input:
       label_list : a list of labeled 'sentences', each element contains a list of indices corresponding to tags
       pred:  pred is the list of predictions,
    Output: accuracy
    """
    if len(label_list) != len(pred):
        print('number of sentences mismatch!')
        return None

    acc = 0.0
    total = 0.0
    for i in range(len(label_list)):
        if len(label_list[i]) != len(pred[i]):
            print('length mismatch on sentence {}'.format(i))
            return None
        total += len(label_list[i])
        acc += sum([1 if label_list[i][j] == pred[i][j]
                    else 0 for j in range(len(label_list[i]))])
    return acc / total


def task_1():
    """
    Hint: This function is fully implemented. Just call it and report your result
    """
    # Test each model given labeled data
    # load the count from training corpus
    word_list, label_list, vocmap, tagmap = load_train_data()
    x = split_data_and_get_counts(1.0, word_list, label_list, vocmap, tagmap)
    t, tw, tpw, tnw, unlabel_word_list, unlabel_label_list = x
    # estimate the parameters
    pt, ptw, ptpw, ptnw = M_step(np.zeros_like(t), np.zeros_like(tw),
                                 np.zeros_like(tpw), np.zeros_like(tnw),
                                 t, tw, tpw, tnw)

    # load the testing data
    word_list, label_list = load_test_data(vocmap, tagmap)

    # predict using each model and evaluate
    pred = predict_A(word_list, pt, ptw, ptpw, ptnw)
    print('Model A accuracy: {}'.format(evaluate(label_list, pred)))

    pred = predict_B(word_list, pt, ptw, ptpw, ptnw)
    print('Model B accuracy: {}'.format(evaluate(label_list, pred)))

    pred = predict_C(word_list, pt, ptw, ptpw, ptnw)
    print('Model C accuracy: {}'.format(evaluate(label_list, pred)))


def task_2():
    """
    Hint: This function is fully implemented. Just call it and report your result
    You will get:
        - 1. The accuracy trained only on the labeled data
        - 2. The log-likelihood and model accuracy after each iteration
    """
    task_em(0.5)


def task_3():
    """
    Hint: This function is fully implemented. Just call it and report your result
    You will get:
        - 1. The accuracy trained only on the labeled data
        - 2. The log-likelihood and model accuracy after each iteration
    """
    task_em(0.01)


def task_em(ratio):
    """
    Run EM algorithm
    Input:
        ratio: the split ratio of labeled and unlabled data; used to compute weight of real counts
    Output:
        Print Log-likelihood and model accuracy for each iteration using models A, B, and C
    """
    word_list, label_list, vocmap, tagmap = load_train_data()

    print('------------------')
    print('Labeled: {}\nUnlabeled: {}'.format(ratio, 1 - ratio))
    print('------------------')

    t, tw, tpw, tnw, unlabel_word_list, unlabel_label_list = split_data_and_get_counts(
        ratio, word_list, label_list, vocmap, tagmap)

    # try different models
    estep_func = [E_step_A, E_step_B, E_step_C]
    likelihood_func = [likelihood_A, likelihood_B, likelihood_C]
    predict_func = [predict_A, predict_B, predict_C]
    name = ['A', 'B', 'C']

    for m in range(len(name)):
        print('Use model ' + name[m] + ':')
        # estimate on labeled data only
        pt, ptw, ptpw, ptnw = M_step(
            np.zeros(
                t.shape), np.zeros(
                tw.shape), np.zeros(
                tpw.shape), np.zeros(
                    tnw.shape), t, tw, tpw, tnw)
        pred = predict_func[m](unlabel_word_list, pt, ptw, ptpw, ptnw)
        print('Model accuracy on labeled data: {}'.format(evaluate(unlabel_label_list, pred)))

        # use the uniform distribution as initialization
        pt, ptw, ptpw, ptnw = M_step(np.zeros_like(t), np.zeros_like(tw),
                                     np.zeros_like(tpw), np.zeros_like(tnw),
                                     np.ones_like(t), np.ones_like(tw),
                                     np.ones_like(tpw), np.ones_like(tnw))

        # run EM
        max_iter = 4
        E_step = estep_func[m]
        likelihood = likelihood_func[m]
        for iter in range(max_iter):
            et, etw, etpw, etnw = E_step(pt, ptw, ptpw, ptnw, unlabel_word_list)
            pt, ptw, ptpw, ptnw = M_step(et, etw, etpw, etnw, t, tw, tpw, tnw)
            l = likelihood(pt, ptw, ptpw, ptnw, unlabel_word_list, t, tw, tpw, tnw)
            pred = predict_func[m](unlabel_word_list, pt, ptw, ptpw, ptnw)
            print('Iter {}, Log-likelihood = {}, Model accuracy = {}'
                  .format(iter + 1, l, evaluate(unlabel_label_list, pred)))


def main():
    """
    The three task functions take care of loading the data and
    running EM on the splits specified in the homework spec
    """
    print('TASK_1*******************************')
    task_1()
    print('TASK_1*******************************')
    print('TASK_2*******************************')
    task_2()
    print('TASK_2*******************************')
    print('TASK_3*******************************')
    task_3()
    print('TASK_3*******************************')


if __name__ == '__main__':
    main()
