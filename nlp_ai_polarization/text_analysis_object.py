'''
Jack Roberge
April 7, 2025
'''

# import libraries
import re
from collections import defaultdict, Counter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
import textstat


class ComparativeTextAnalysis:

    def __init__(self):
        """ Contructor """
        self.data = defaultdict(dict)
        self.stop_words = set()

    def load_stop_words(self, stopfile):
        """Load stop words from a file into set  if not pre-defined """
        with open(stopfile, 'r') as f:
            self.stop_words = set(f.read().split())

    def load_text(self, filename, label=None, parser=None):
        """ Register a document with the framework and
        store data extracted from the document to be used
        later in visualizations """

        results = self.simple_text_parser(filename) # default
        if parser is not None:
            results = parser(filename)

        if label is None:
            label = filename

        for k, v in results.items():
            self.data[k][label] = v

    def simple_text_parser(self, filename):
        """Parse basic text and filter out stop words"""

        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        # Find flesch grade complexity score of all the text
        complexity_score = textstat.flesch_kincaid_grade(text)

        # Clean + tokenize
        clean = re.sub(r'[^\w\s]', '', text.lower())
        words = clean.split()

        # Filter stop words
        filtered_words = [w for w in words if w not in self.stop_words]

        # Word count and word length
        wordcount = Counter(filtered_words)
        avg_word_length = sum(len(w) for w in filtered_words) / len(filtered_words) if filtered_words else 0

        # Sentiment analysis using nltk VADER
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        sentiment = {"negative": scores["neg"], "positive": scores["pos"]}

        # Create and store corpus data
        results = {'wordcount': wordcount,
                   'numwords': len(filtered_words),
                   'avg_word_length': avg_word_length,
                   'sentiment': sentiment,
                   'complexity': complexity_score}

        return results

    def wordcount_sankey(self, word_list=None, files_of_interest=None, k=5):
        ''' sankey function to create a sankey diagram for the files of interest
        using either the word_list provided or the k-most frequent words in each file '''
        source = []
        target = []
        value = []

        # Create a list of text labels (sources)
        all_text_labels = list(self.data['wordcount'].keys())

        # Filter files to only include files of interest
        text_labels = files_of_interest if files_of_interest else all_text_labels

        # Get top-k words per text OR use provided word_list
        word_freqs = {}
        all_words = set()

        for text in text_labels:
            wordcount = self.data['wordcount'][text]

            if word_list:
                selected_words = {word: wordcount[word] for word in word_list if word in wordcount}
            else:
                selected_words = dict(wordcount.most_common(k))

            word_freqs[text] = selected_words
            all_words.update(selected_words.keys())

        # Create a list of word labels (targets)
        word_labels = list(all_words)

        # Map text labels to word labels
        for text, words in word_freqs.items():
            text_index = text_labels.index(text)
            for word, count in words.items():
                word_index = len(text_labels) + word_labels.index(word)
                source.append(text_index)
                target.append(word_index)
                value.append(count)

        # Define node labels
        labels = text_labels + word_labels

        # Create Sankey
        fig = go.Figure(go.Sankey(
            node=dict(
                pad=15, thickness=20, line=dict(color="black", width=0.5),
                label=labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        ))

        fig.update_layout(title_text="Text-to-Word Sankey Diagram", font_size=10)
        fig.show()



    def sentiment_plot(self, files_of_interest=None):
        """
        Create subplot with a bar chart for each file displaying
        positive and negative sentiment scores.
        """

        # include only files of interest
        all_sentiments = self.data['sentiment']
        sentiments = {k: v for k, v in all_sentiments.items() if not files_of_interest or k in files_of_interest}

        # determine suplot sizes
        num_articles = len(sentiments)
        rows = math.ceil(num_articles / 2)

        # Create subplot with a barchart for each article
        fig = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=list(sentiments.keys()),
            horizontal_spacing=0.15
        )

        # Iterate with an index to calc row and col position
        for i, (label, sentiment) in enumerate(sentiments.items()):
            row = i // 2 + 1
            col = i % 2 + 1
            fig.add_trace(
                go.Bar(
                    x=["Negative", "Positive"],
                    y=[sentiment["negative"], sentiment["positive"]],
                    name=label
                ),
                row=row, col=col
            )
            fig.update_yaxes(title_text="Sentiment Score", row=row, col=col)
            fig.update_xaxes(title_text="Sentiment", row=row, col=col)

        # Plot layout/format
        fig.update_layout(
            height=400 * rows,
            title={
                "text": "<b>Overall Sentiment Analysis</b>",
                "x": 0.5,
                "xanchor": "center"
            },
            showlegend=False,
            margin=dict(l=0, r=0, b=50, t=100)
        )
        fig.show()

    def avg_word_length_plot(self, files_of_interest=None):
        """Create a single plot with average word length for selected
        files"""

        # include only average word lengths from files of interest
        avg_word_lengths = {}

        for label, value in self.data['avg_word_length'].items():
            if not files_of_interest or label in files_of_interest:
                avg_word_lengths[label] = value

        # graph files and their average word lengths
        texts = list(avg_word_lengths.keys())
        values = [avg_word_lengths[text] for text in texts]

        fig = go.Figure(
            data=go.Bar(
                x=texts,
                y=values,
                name="Average Word Length",
                marker_color='blue'
            )
        )

        fig.update_layout(
            title_text="Average Word Length by Document",
            xaxis_title="Documents",
            yaxis_title="Average Word Length",
            height=500
        )

        fig.show()

    def complexity_plot(self, files_of_interest=None):
        """Create a single plot with Flesch Reading Grade score
        (text complexity) for each file."""

        # include only complexity scores from files of interest
        complexity_scores = {}

        for label, value in self.data['complexity'].items():
            if not files_of_interest or label in files_of_interest:
                complexity_scores[label] = value

        # graph files and their complexity scores (flesch reading grades)
        texts = list(complexity_scores.keys())
        values = [complexity_scores[text] for text in texts]

        fig = go.Figure(
            data=go.Bar(
                x=texts,
                y=values,
                name="Complexity Score",
                marker_color='green'
            )
        )

        fig.update_layout(
            title_text="Text Complexity (Flesch Grade Level) by Document",
            xaxis_title="Documents",
            yaxis_title="Complexity Score",
            height=500
        )

        fig.show()