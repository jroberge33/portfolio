'''
Jack Roberge
 April 7, 2025
'''

# import text analysis object and stopwords
from text_analysis_object import ComparativeTextAnalysis
from nltk.corpus import stopwords

def main():

    # define models used and prompt types in file naming convention
    models = ['chat', 'claude3.7', 'perplexity', 'deepseek', 'gemini', 'grok']
    file_endings = ['base_simple', 'base_verbose', 'conservative_slant_simple',
                    'conservative_slant_verbose', 'liberal_slant_simple', 'liberal_slant_verbose']

    # create model and prompt type labels for output graphs
    model_labels = {
        'chat': 'ChatGPT',
        'claude3.7': 'Claude',
        'perplexity': 'Perplexity',
        'deepseek': 'DeepSeek',
        'gemini': 'Gemini',
        'grok': 'Grok'
    }
    file_labels = {
        'base_simple': 'Unbiased',
        'base_verbose': 'Unbiased (Verbose)',
        'conservative_slant_simple': 'Conservative',
        'conservative_slant_verbose': 'Conservative (Verbose)',
        'liberal_slant_simple': 'Liberal',
        'liberal_slant_verbose': 'Liberal (Verbose)'
    }

    # create text analysis object and load stopwords
    cta = ComparativeTextAnalysis()
    cta.stop_words = set(stopwords.words('english'))

    # load models into text analysis object
    all_labels = {}

    for model in models:
        for ending in file_endings:
            filename = f'{model}_{ending}.txt'
            model_name = model_labels.get(model, model.capitalize())
            label_suffix = file_labels.get(ending, ending.replace('_', ' ').capitalize())
            label = f"{model_name}: {label_suffix}"
            all_labels[(model, ending)] = label
            cta.load_text(filename, label)


    # graph only liberally slanted liberal prompts in sankey to maintain readability
    sankey_files_of_interest = [
        all_labels[(model, ending)]
        for model in models
        for ending in ['liberal_slant_simple']
    ]
    # create sankey for only 4 top words per file
    cta.wordcount_sankey(k=4, files_of_interest=sankey_files_of_interest)

    # graph all ideological slants for all models with only verbose prompt complexities
    sentiment_files_of_interest = [
        all_labels[(model, ending)]
        for model in models
        for ending in ['base_verbose', 'conservative_slant_verbose', 'liberal_slant_verbose']
    ]
    cta.sentiment_plot(files_of_interest=sentiment_files_of_interest)

    # graph all complexities for all models with only unbiased ideological prompt slants
    complexity_files_of_interest = [
        all_labels[(model, ending)]
        for model in models
        for ending in ['base_simple', 'base_verbose']
    ]
    # graph avg word length across model prompt complexities and flesch grade level
    cta.avg_word_length_plot(files_of_interest=complexity_files_of_interest)
    cta.complexity_plot(files_of_interest=complexity_files_of_interest)

if __name__ == '__main__':
    main()