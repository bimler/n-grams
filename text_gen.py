"""This program generates random text based on n-grams
calculated from sample text.

Author: Nathan Sprague and Brady Imler
Date: 1/10/20
Modified: 1/27/20

"""

# Honor code statement (if you received help from an outside source):
# This work complies with the JMU honor code

import random
import string


def text_to_list(file_name):
    """ Converts the provided plain-text file to a list of words.  All
    punctuation will be removed, and all words will be converted to
    lower-case.

    Argument:
        file_name - A string containing a file path.
    Returns
        A list containing the words from the file.
    """
    handle = open(file_name, 'r')
    text = handle.read().lower()
    text = text.translate(
        str.maketrans(string.punctuation,
                      " " * len(string.punctuation)))
    return text.split()


def select_random(distribution):
    """
    Select an item from the the probability distribution
    represented by the provided dictionary.

    Example:
    >>> select_random({'a':.9, 'b':.1})
    'a'
    """

    # Make sure that the probability distribution has a sum close to 1.
    assert abs(sum(distribution.values()) - 1.0) < .000001, \
        "Probability distribution does not sum to 1! " + str(abs(sum(distribution.values()) - 1.0))

    r = random.random()
    total = 0
    for item in distribution:
        total += distribution[item]
        if r < total:
            return item

    assert False, "Error in select_random!"


def counts_to_probabilities(counts):
    """ Convert a dictionary of counts to probabilities.

    Argument:
       counts - a dictionary mapping from items to integers

    Returns:
       A new dictionary where each count has been divided by the sum
       of all entries in counts.

    Example:

    >>> counts_to_probabilities({'a':9, 'b':1})
    {'a': 0.9, 'b': 0.1}

    """
    probabilities = {}
    total = 0
    for item in counts:
        total += counts[item]
    for item in counts:
        probabilities[item] = counts[item] / float(total)
    return probabilities


def calculate_unigrams(word_list):
    """ Calculates the probability distribution over individual words.

    Arguments:
       word_list - a list of strings corresponding to the
                   sequence of words in a document. Words must
                   be all lower-case with no punctuation.
    Returns:
       A dictionary mapping from words to probabilities.

    Example:

    >>> u = calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    >>> print u
    {'i': 0.4, 'am': 0.2, 'think': 0.2, 'therefore': 0.2}

    """
    unigrams = {}
    for word in word_list:
        if word in unigrams:
            unigrams[word] += 1
        else:
            unigrams[word] = 1
    return counts_to_probabilities(unigrams)


def random_unigram_text(unigrams, num_words):
    """Generate a random sequence according to the provided probabilities.

    Arguments:
       unigrams -   Probability distribution over words (as returned by the
                    calculate_unigrams function).
       num_words -  The number of words of random text to generate.

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:

    >>> u = calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    >>> random_unigram_text(u, 5)
    'think i therefore i i'

    """
    result = ""
    for i in range(num_words):
        next_word = select_random(unigrams)
        result += next_word + " "
    return result.rstrip()


def calculate_bigrams(word_list):
    """Calculates, for each word in the list, the probability distribution
    over possible subsequent words.

    This function returns a dictionary that maps from words to
    dictionaries that represent probability distributions over
    subsequent words.

    Arguments:
       word_list - a list of strings corresponding to the
                   sequence of words in a document. Words must
                   be all lower-case with no punctuation.

    Example:

    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think', 'am', 'therefore', 'think'])
    >>> print b
    {'i':  {'am': 0.25, 'think': 0.75},
     None: {'i': 1.0},
     'am': {'i': 1.0},
     'think': {'i': 0.5, 'therefore': 0.5},
     'therefore': {'i': 1.0}}

    Note that None stands in as the predecessor of the first word in
    the sequence.

    Once the bigram dictionary has been obtained it can be used to
    obtain distributions over subsequent words, or the probability of
    individual words:

    >>> print b['i']
    {'am': 0.25, 'think': 0.75}

    >>> print b['i']['think']
    .75

    """
    # add initial None to the list, followed by the probability of 1 for the next word
    index = 0
    ret_dict = {None: {word_list[0]: 1.0}}
    count_dict = {None: 1}
    occurrences = None
    for word in word_list:
        # keeps count of total occurences, for calculating probability
        if word in count_dict and index is not len(word_list) - 1:
            count_dict[word] = count_dict[word] + 1
        elif index is not len(word_list) - 1:
            count_dict[word] = 1
        # adds new words to dictionary and increments the counts
        if word in ret_dict and index < len(word_list) - 1:
            inner_dict = ret_dict.get(word)
            if word_list[index + 1] in inner_dict:
                inner_dict[word_list[index + 1]] = inner_dict[word_list[index + 1]] + 1.0
            else:
                inner_dict[word_list[index + 1]] = 1.0
        elif index < len(word_list) - 1:
            ret_dict[word] = {word_list[index + 1]: 1.0}
        index = index + 1
    # divides the counts of each individual, by the total count for each word, for probability
    for elem in ret_dict:
        inner = ret_dict[elem]
        for inner_elem in inner:
            inner[inner_elem] = inner[inner_elem] / count_dict[elem]
    return ret_dict


def calculate_trigrams(word_list):
    """Calculates, for each adjacent pair of words in the list, the
    probability distribution over possible subsequent words.

    The returned dictionary maps from two-word tuples to dictionaries
    that represent probability distributions over subsequent
    words.

    Example:

    >>> b = calculate_trigrams(['i', 'think', 'therefore', 'i', 'am',\
                                'i', 'think', 'i', 'think'])
    >>> print b
    {('think', 'i'): {'think': 1.0},
    ('i', 'am'): {'i': 1.0},
    (None, None): {'i': 1.0},
    ('therefore', 'i'): {'am': 1.0},
    ('think', 'therefore'): {'i': 1.0},
    ('i', 'think'): {'i': 0.5, 'therefore': 0.5},
    (None, 'i'): {'think': 1.0},
    ('am', 'i'): {'think': 1.0}}
    """
    # add initial None to the list, followed by the probability of 1 for the next word
    ret_dict = {
        (None, None): {word_list[0]: 1.0},
        (None, word_list[0]): {word_list[1]: 1.0}
    }
    occurrences = None
    count_dic = {
        (None, word_list[0]): 1,
        (None, None): 1
    }
    index = 0
    for word in word_list:
        if index < len(word_list) - 2:
            working = (word, word_list[index + 1])
            # incrementing count for each (word, word)
            if count_dic.get(working) is not None:
                count_dic[working] = count_dic.get(working) + 1.0
            else:
                count_dic[working] = 1.0
            # adds words to dictionary and increments counts
            if ret_dict.get(working) is not None:
                inner = ret_dict[working]
                if inner.get(word_list[index + 2]) is not None:
                    inner[word_list[index + 2]] = inner[word_list[index + 2]] + 1.0
                else:
                    inner[word_list[index + 2]] = 1.0
                ret_dict[working] = inner
            else:
                ret_dict[working] = {word_list[index + 2]: 1.0}
        index = index + 1
    # adjusting count by dividing by total count for each word
    for elem in ret_dict:
        inner = ret_dict[elem]
        for inner_elem in inner:
            inner[inner_elem] = inner[inner_elem] / count_dic[elem]

    return ret_dict


def random_bigram_text(first_word, bigrams, num_words):
    """Generate a random sequence of words following the word pair
    probabilities in the provided distribution.

    Arguments:
       first_word -          This word will be the first word in the
                             generated text.
       bigrams -             Probability distribution over word pairs
                             (as returned by the calculate_bigrams function).
       num_words -           Length of the generated text (including the
                             provided first word)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:
    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think'])
    >>> random_bigram_text('think', b, 5)
    'think i think therefore i'

    >>> random_bigram_text('think', b, 5)
    'think therefore i think therefore'

    """
    text = ""
    text = text + first_word
    previous = first_word
    for i in range(num_words - 1):
        dict = bigrams.get(previous)
        previous = select_random(dict)
        text = text + " " + previous
    return text


def random_trigram_text(first_word, second_word, bigrams, trigrams, num_words):
    """Generate a random sequence of words according to the provided
    bigram and trigram distributions.

    By default, each new word will be generated using the trigram
    distribution.  The bigram distribution will be used when a
    particular word pair does not have a corresponding trigram.

    Arguments:
       first_word -          The first word in the generated text.
       second_word -         The second word in the generated text.
       bigrams -             bigram probabilities (as returned by the
                             calculate_bigrams function).
       trigrams -            trigram probabilities (as returned by the
                             calculate_bigrams function).
       num_words -           Length of the generated text (including the
                             provided words)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    """
    text = ""
    text = first_word + " " + second_word
    previous2 = first_word
    previous1 = second_word
    for i in range(num_words - 2):
        dict = trigrams.get((previous2, previous1))
        if dict is not None:
            previous2 = previous1
            previous1 = select_random(dict)
            text = text + " " + previous1
        else:
            # no trigram
            dict2 = bigrams.get(previous1)
            previous2 = previous1
            previous1 = select_random(dict2)
            text = text + " " + previous1
    return text


def unigram_main():
    """ Generate text from Huck Fin unigrams."""
    words = text_to_list('huck.txt')
    unigrams = calculate_unigrams(words)
    print(random_unigram_text(unigrams, 500))


def bigram_main():
    """ Generate text from Huck Fin bigrams."""
    words = text_to_list('huck.txt')
    bigrams = calculate_bigrams(words)
    print(random_bigram_text('the', bigrams, 500))


def trigram_main():
    """ Generate text from Huck Fin trigrams."""
    words = text_to_list('huck.txt')
    bigrams = calculate_bigrams(words)
    trigrams = calculate_trigrams(words)
    print(random_trigram_text('there', 'is', bigrams, trigrams, 500))


if __name__ == "__main__":
    # You can insert testing code here, or switch out the main method
    # to try bigrams or trigrams.

    unigram_main()
    print("\n")
    bigram_main()
    print("\n")
    trigram_main()
