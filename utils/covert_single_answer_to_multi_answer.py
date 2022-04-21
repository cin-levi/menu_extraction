import json
import uuid
import tqdm


def generate_question_list(data_file, questions_path):
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    all_questions = []
    for sample in data['data']:
        for paragraph in sample['paragraphs']:
            for qa in paragraph['qas']:
                all_questions.append(qa['question'])
    all_questions = sorted(list(set(all_questions)))
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(all_questions, f, ensure_ascii=False)
    return all_questions


def convert_single_to_multi_answer_format(original_train_data_or_file, all_questions, output_file):
    if isinstance(original_train_data_or_file, dict):
        data = original_train_data_or_file['data']
    else:
        with open(original_train_data_or_file, 'r', encoding='utf-8') as f:
            data = json.load(f)['data']

    new_data = {'data': []}

    num_original_questions = 0
    num_added_questions = 0

    for x in data:
        new_doc = {'title': x['title'], 'paragraphs': []}

        for paragraph in x['paragraphs']:
            all_qas = {}
            context = paragraph['context']
            qas = paragraph['qas']
            for qa in qas:
                qa['is_augmentation'] = False
                question = qa['question']
                if question not in all_qas:
                    all_qas[question] = qa
                else:
                    all_qas[question]['answers'].extend(qa['answers'])
                num_original_questions += 1
                # all_questions.append(question)
            all_qas = [all_qas[q] for q in all_qas]
            all_questions_in_paragraph = [x['question'] for x in all_qas]
            assert len(list(set(all_questions_in_paragraph))) == len(all_questions_in_paragraph)
            for question in all_questions:
                if question not in all_questions_in_paragraph:
                    num_added_questions += 1
                    all_qas.append(
                        {'question': question, 'is_impossible': True, 'answers': [], 'id': str(uuid.uuid4())})
            new_paragraph = {'context': paragraph['context'], 'qas': all_qas, 'uid': str(uuid.uuid4())}
            if 'position' in paragraph:
                new_paragraph['position'] = paragraph['position']
            new_doc['paragraphs'].append(new_paragraph)
        new_data['data'].append(new_doc)
    print('saving to ', output_file)
    print(f"original: {num_original_questions}")
    print(f"added: {num_added_questions}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False)


if __name__ == '__main__':
    # data_path = 'D:\\FastQA\\original_data\\address\\train.json'
    # questions_path = 'D:\\FastQA\\data\\address\\question_list.json'
    # generate_question_list(data_path, questions_path)
    from processors import prj_path
    original_train_data_file = prj_path + '/data\\full.json'
    questions_path = prj_path + '/data\\question_list.json'
    output_file = prj_path + '/data\\full.json'
    with open(questions_path, 'r', encoding='utf-8') as f:
        question_dict = json.load(f)
    all_questions = sorted([x for x in question_dict])
    convert_single_to_multi_answer_format(original_train_data_file, all_questions, output_file)
