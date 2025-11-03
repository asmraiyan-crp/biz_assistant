from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class Sentiment:
    def __init__(self):
        """
        Initializes the sentiment analyzer.
        Uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
        which is specifically tuned for sentiment and understands context
        like "not good".
        """
        try:
            self.analyzer = SentimentIntensityAnalyzer()
        except Exception as e:
            print(f"Failed to load VADER analyzer: {e}")
            self.analyzer = None

    def score(self, text):
        """
        Scores the sentiment of a given text.

        Args:
            text (str): The input text to analyze.

        Returns:
            float: A compound sentiment score between -1.0 (most negative)
                   and 1.0 (most positive). Returns 0.0 for invalid input.
        """
        if not self.analyzer or not isinstance(text, str) or not text.strip():
            return 0.0

        # Get sentiment scores
        scores = self.analyzer.polarity_scores(text)

        # The 'compound' score is a normalized, weighted score
        return float(scores['compound'])