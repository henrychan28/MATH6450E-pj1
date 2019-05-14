from __future__ import division

import pickle
import math

import numpy as np
from scipy.special import digamma, gammaln


def get_preprocessed_data(m=100):
    with open('normalized_tokenized_data.pkl', 'rb') as f:
        documents = pickle.load(f)[:m]
    vocabulary = set()
    for datum in documents:
        vocabulary.update(datum)
    vocabulary = list(vocabulary)
    return documents, vocabulary


def doc_to_match_given_vocabulary(doc, word):
    return (np.array(doc) == word).astype(int)


def initialization(documents, K, M, V, alpha_init=1.0, beta_init=0, dtype=np.float32):
    alpha = np.repeat(alpha_init, K) * 50 / K
    beta = np.full([K, V], beta_init, dtype=dtype)
    for k in range(0, K):
        unnormalized_probability = np.random.uniform(0, 1, V)
        beta[k, :] = unnormalized_probability / np.sum(unnormalized_probability)
    gamma = np.full([M, K], 0, dtype=dtype)
    phi = []
    for m, doc in enumerate(documents):
        gamma[m, :] = alpha + len(doc) / K
        phi += [np.full([len(doc), K], 1 / K)]
    return alpha, beta, phi, gamma


def E_step(alpha, beta, phi, gamma, documents, vocabulary, K, M, V):
    print("E-Step")
    for m, doc in enumerate(documents):
        converge = False
        converge_iteration = 0
        while not converge:
            phi_old = np.array(phi[m], copy=True)
            gamma_old = np.array(gamma[m], copy=True)
            for n in range(len(doc)):
                for i in range(K):
                    word_idx = vocabulary.index(doc[n])
                    phi[m][n][i] = beta[i, word_idx] * math.exp(digamma(gamma[m, i]) - digamma(np.sum(gamma[m, :])))
                if np.sum(phi[m][n]) == 0:
                    print("beta: {0} | digamma: {1} | digamma_sum: {2} | word: {3}".format(beta[i, word_idx],
                                                                                           digamma(gamma[m, i]),
                                                                                           digamma(np.sum(gamma[m])),
                                                                                           vocabulary[word_idx]))
                phi[m][n] = phi[m][n] / np.sum(phi[m][n])  # Normalization
            gamma[m] = alpha + np.sum(phi[m], axis=0)
            phi_error = np.linalg.norm(phi_old - phi[m])
            gamma_error = np.linalg.norm(gamma_old - gamma[m])
            if phi_error < 1e-3 and gamma_error < 1e-3:
                converge = True
                print("Document: {0} | Iteration: {1} | "
                      "Phi_err: {2} | Gamma_err: {3}".format(m, converge_iteration,
                                                             phi_error, gamma_error))
            converge_iteration += 1
    return phi, gamma


def M_step(alpha, beta, phi, gamma, documents, vocabulary, K, M, V):
    print("M-Step")
    beta_old = np.array(beta, copy=True)
    beta = np.zeros([K, V], dtype=np.float32)
    for m, doc in enumerate(documents):
        for j in range(V):
            w_j = doc_to_match_given_vocabulary(doc, vocabulary[j])
            beta[:, j] += np.dot(w_j, phi[m])

    beta = beta / np.sum(beta, axis=1)[:, None]
    if np.any(beta == 0.0):
        print("Skip Updating Beta as there is an element in beta equals to zero.")
        return alpha, beta_old

    alpha = alpha
    return alpha, beta


def likelihood_per_document(alpha, beta, phi, gamma, doc, vocabulary, K, M, V):
    gamma_total = -np.log(gammaln(np.sum(gamma))) + np.sum(np.log(gammaln(gamma)))
    alpha_total = np.log(gammaln(np.sum(alpha))) - np.sum(np.log(gammaln(alpha)))
    phi_entropy = -np.sum(phi * np.log(phi))
    phi_digamma_gamma = np.dot(phi, digamma(gamma) - digamma(np.sum(gamma))).sum()
    alpha_digamma_gamma = np.dot(alpha - 1, digamma(gamma) - digamma(np.sum(gamma))).sum()
    gamma_digamma_gamma = np.dot(gamma - 1, digamma(gamma) - digamma(np.sum(gamma))).sum()
    word_matrix = np.zeros([len(doc), len(vocabulary)])
    phi_log_beta = np.dot(phi, np.log(beta))
    if not (np.Infinity in phi_log_beta or -np.Infinity in phi_log_beta):
        for idx, word in enumerate(doc):
            word_matrix[idx, vocabulary.index(word)] = 1
        phi_word_log_beta = np.sum(np.multiply(phi_log_beta, word_matrix))
    else:
        phi_word_log_beta = 0
    return gamma_total + alpha_total + phi_entropy + phi_digamma_gamma + \
           alpha_digamma_gamma + gamma_digamma_gamma + phi_word_log_beta


def get_likelihood(alpha, beta, phi, gamma, documents, vocabulary, K, M, V):
    likelihood_total = 0
    for m, doc in enumerate(documents):
        likelihood_total += likelihood_per_document(alpha, beta, phi[m], gamma[m], doc, vocabulary, K, M, V)
    return likelihood_total


def variational_inference(alpha, beta, phi, gamma, documents, vocabulary, K, M, V):
    likelihood = 0
    likelihood_previous = -1e3
    iteration = 0
    likelihood = get_likelihood(alpha, beta, phi, gamma, documents, vocabulary, K, M, V)
    print("Iteration: {0} | likelihood: {1} | likelihood_previous: {2}".format(-1, likelihood, 'NA'))
    while abs(likelihood - likelihood_previous) > 1e-4:
        likelihood_previous = likelihood
        phi, gamma = E_step(alpha, beta, phi, gamma, documents, vocabulary, K, M, V)
        alpha, beta = M_step(alpha, beta, phi, gamma, documents, vocabulary, K, M, V)
        likelihood = get_likelihood(alpha, beta, phi, gamma, documents, vocabulary, K, M, V)
        print("Iteration: {0} | likelihood: {1} | likelihood_previous: {2}".format(iteration, likelihood,
                                                                                   likelihood_previous))
        iteration += 1
    return alpha, beta, phi, gamma


def LDA(documents, vocabulary, K=4):
    M = len(documents)
    V = len(vocabulary)
    alpha, beta, phi, gamma = initialization(documents, K, M, V)
    alpha, beta, phi, gamma = \
        variational_inference(alpha, beta, phi, gamma, documents, vocabulary, K, M, V)
    return alpha, beta, phi, gamma


if __name__ == "__main__":
    m = 100
    K = 4
    documents, vocabulary = get_preprocessed_data(m=m)
    print("Number of Vocabulary: {0}".format(len(vocabulary)))
    LDA(documents, vocabulary)
