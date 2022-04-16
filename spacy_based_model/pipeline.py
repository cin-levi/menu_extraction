import json

from tqdm import tqdm
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

nlp = spacy.load("en_core_web_md")

# matcher patterns to get the right date format.
matcher = Matcher(nlp.vocab)
patterns = [[{"TEXT": {"REGEX": "^(19|20)\d{2}$"}}]]
matcher.add("match_id", patterns)

# Preprocessing changes
changes = {0: [' NV ', ' NV  $'],
           1: [' nv ', ' nv  $'],
           2: ['\n\n$', ' $'],
           3: ['\n\n', '\n'],
           4: ['\n  \n  \n  \n  \n', ' '],
           5: ['\n$', ' $'],
           6: ['**$', ' $'],
           7: ['$ ', '$'],
           8: ['  $', ' $'],
           9: [' $', '  $'],
           10: ['\n', '\n '],
           11: [' ', '   ']}

# punctuations
punctuations = '!"#%&\*+:;<=>?@[]^`{|}~'


class WineNerPipeline:

    def __init__(self, pages_text, changes, punctuation, model, matcher):
        self.pages_text = pages_text
        self.changes = changes
        self.punctuation = punctuation
        self.model = model
        self.matcher = matcher

    def __call__(self):

        """
         WineNerPipeline : text_preprocessing >> data_extractor >> save_data.
        """

        processedTexts = self.text_preprocessing()
        allMenus, datesPrices, winesSeq, wineCoordinates, preds = self.data_extractor(processedTexts)
        return allMenus, datesPrices, winesSeq, wineCoordinates, preds

    def text_preprocessing(self):
        """
        removing punctuation and special characters

        Returns:
          processedTexts (:list): Text after preprocessing for each page.
        """
        processedTexts = []
        for text in self.pages_text:
            newText = "".join([i for i in text if i not in self.punctuation])
            for item in self.changes:
                newText = newText.replace(changes[item][0], changes[item][1])

            processedTexts.append(newText)

        return processedTexts

    def data_extractor(self, processedTexts):

        """
        Args :
          processedTexts : list of pages text after preprocessing.
        Returns :
          allMenus : list of potentia menus for each page.
          datesPrices : list of pairs [date, price]
        """
        allMenus, datesPrices, winesSeq, wineCoordinates = [], [], [], []
        labels = []

        for text in tqdm(processedTexts):

            menu, coordinates, wines, elements = [''], [''], [''], [[None, None]]
            menu_idx = 0
            doc = self.model(text)

            for ent in doc.ents:

                date, price = None, None  # Initialization with None

                # searching for prices or dates that respect the matcher format.

                if ent.label_ == 'MONEY' or (ent.label_ == "DATE" and self.matcher(ent)):
                    # left_idx = 3  # searching moving backward
                    left_idx = min(3, ent.start)  # searching moving backward
                    right_idx = 0  # seaching forward

                    if ent.label_ == "DATE":
                        right_idx = 1

                    # this while loops defines how much steps we need to get the right sentence.
                    while ent.start - left_idx > 0 and '\n' not in doc[ent.start - left_idx].text:
                        left_idx += 1

                    while ent.end + right_idx < len(doc) and '\n' not in doc[ent.end + right_idx].text:
                        right_idx += 1
                        # avoiding IndexOutOfBounds
                        if (ent.end + right_idx) >= (len(doc) - 1):
                            right_idx = 1
                            break

                    if (ent.label_ == "DATE" and matcher(ent)):
                        date = ent.text
                        price = doc[
                                ent.end: ent.end + right_idx].text  # sentence = "... date .. price\n" ==> price = "..price\n"
                        # Ner spacy models requires characters coordinates for training
                        wine = doc[ent.start - left_idx: ent.start].text
                        wine_span = Span(doc, ent.start - left_idx, ent.start)
                        wine_start_end = (
                            wine_span.start_char, wine_span.end_char)  # (wine start character, wine end character)

                    # ent.label_=="PRICE"
                    else:
                        price = doc[ent.start - 1: ent.end + right_idx].text
                        wine = doc[ent.start - left_idx: ent.start - 1].text  # -1 to exclude $ symbol.

                        wine = doc[ent.start - left_idx: ent.start - 1].text
                        try:
                            wine_span = Span(doc, ent.start - left_idx, ent.start - 1)
                            wine_start_end = (
                                wine_span.start_char, wine_span.end_char)  # (wine start character, wine end character)
                        except:
                            wine_start_end = (0,0)

                    spansText = doc[ent.start - left_idx: ent.end + right_idx].text  #

                    # This block help us to avoid duplications and get the best sentence that contains all the availible information

                    if (menu[menu_idx] in spansText) and (len(menu[menu_idx]) < len(spansText)):

                        menu[menu_idx] = spansText
                        coordinates[menu_idx] = wine_start_end
                        wines[menu_idx] = wine
                        if date and not elements[menu_idx][0]:
                            elements[menu_idx][0] = date
                        if price and not elements[menu_idx][1]:
                            elements[menu_idx][1] = price

                    elif (spansText in menu[menu_idx]) and (len(menu[menu_idx]) >= len(spansText)):

                        if len(menu[menu_idx]) > len(spansText):
                            wines[menu_idx] = wine
                            coordinates[menu_idx] = wine_start_end


                    else:
                        menu.append(spansText)
                        elements.append([date, price])
                        coordinates.append(wine_start_end)
                        wines.append(wine)
                        menu_idx += 1

            doctext = doc.text
            # print(coordinates)
            try:
                coordinates = sorted(coordinates, key=lambda x: x[0])
            except:
                coordinates = []
            _plabels = []
            current_start = 0
            for coord in coordinates:
                prior = doctext[current_start:coord[0]]
                inner = doctext[coord[0]:coord[1]]
                current_start = coord[1]
                _plabels.extend(['O'] * len(prior.strip().split()))
                inner_splited = inner.strip().split()
                _plabels.extend(['B-w'] + ['I-w'] * (len(inner_splited) - 1))
            last = doctext[current_start:]
            _plabels.extend(['O'] *len(last.strip().split()))
            labels.append(_plabels)


            allMenus.append(menu)
            winesSeq.append(wines)
            datesPrices.append(elements)
            wineCoordinates.append(coordinates)
        return allMenus, datesPrices, winesSeq, wineCoordinates, labels


# TODO: calculate accuracy
# Need to merge text into using the start position

data_folder = 'D:\\menu_extraction\\data\\kfold'
file = 'D:\\menu_extraction\\data\\kfold\\fold_2\\test.json'

with open(file, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']

from sklearn.cluster import DBSCAN

db_scan = DBSCAN(eps=2, min_samples=1)
labels = []
tokens = []
for sample in data:
    page_text = []
    page_labels = []
    for paragraph in sample['paragraphs']:
        context = paragraph['context']
        positions = paragraph['position']
        assert len(context.split()) == len(positions)
        char_to_word_offset = []
        _tokens = []
        word = ''
        for i, c in enumerate(context):
            if c == ' ':
                if len(word) > 0:
                    _tokens.append(word)
                    word = ''
            else:
                word += c
            char_to_word_offset.append(len(_tokens))
        if len(word) > 0:
            _tokens.append(word)
        _labels = ['O'] * len(_tokens)
        answers = []
        for qa in paragraph['qas']:
            question = qa['question']
            for answer in qa['answers']:
                answers.append(
                    (question, answer['answer_start'], answer['answer_start'] + len(answer['text']), answer['text']))
        answers = sorted(answers, key=lambda x: x[1])

        for answer in answers:
            token_indexes = sorted(list(set(char_to_word_offset[answer[1]: answer[2]])))
            entity_labels = ['B-' + answer[0]] + ['I-' + answer[0]] * (len(token_indexes) - 1)
            for i, index in enumerate(token_indexes):
                _labels[index] = entity_labels[i]
        assert len(_tokens) == len(_labels)
        assert len(_tokens) == len(positions)
        # Horizontal merge
        line_label = db_scan.fit_predict([[p[-1]] for p in positions])
        unique_labels = sorted(list(set(line_label)))
        for l in unique_labels:
            indexes = [i for i in range(len(line_label)) if line_label[i] == l]
            page_labels.append([_labels[i] for i in indexes])
            page_text.append(' '.join([_tokens[i] for i in indexes]) + '\n')
    labels.extend(page_labels)
    tokens.extend(page_text)

from seqeval.metrics import classification_report

extractor1 = WineNerPipeline(tokens, changes, punctuations, nlp, matcher)

allMenus, datesPrices, winesSeq, wineCoordinates, preds = extractor1()
print(allMenus)
print(wineCoordinates)
print()
accepted_labels = []
accepted_preds = []
for i in range(len(labels)):
    if len(labels[i]) == len(preds[i]):
        accepted_labels.append(labels[i])
        accepted_preds.append(preds[i])
print("Original: ", len(labels))
print("Accepted: ", len(accepted_labels))
print(classification_report(y_true=accepted_labels, y_pred=accepted_preds))
