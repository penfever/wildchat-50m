from datasets import Features, Value, Sequence

def convert_header_types(features_dict):
    """
    Convert header fields to string type in a features dictionary.
    """
    # Create a new features structure with correct types
    new_features = Features({
        'conversation_hash': Value('string'),
        'timestamp': Value('timestamp[us, tz=UTC]'),
        'conversation': [{
            'content': Value('string'),
            'content_token_ids': Sequence(Value('int64')),
            'country': Value('string'),
            'cumulative_logprob': Value('null'),
            'finish_reason': Value('string'),
            'hashed_ip': Value('string'),
            'header': {
                'accept-language': Value('string'),  # Forced to string type
                'user-agent': Value('string'),       # Forced to string type
            },
            'judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_Factuality_content': Value('string'),
            'judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_Factuality_cumulative_logprob': Value('string'),
            'judgment_meta-llama_Llama-3.1-8B-Instruct_conversation_Factuality_logprob': Value('string'),
            'language': Value('string'),
            'redacted': Value('bool'),
            'role': Value('string'),
            'state': Value('string'),
            'timestamp': Value('timestamp[us, tz=UTC]'),
            'toxic': Value('bool'),
            'turn_identifier': Value('int64')
        }],
        'turn': Value('int64'),
        'language': Value('string'),
        'openai_moderation': [{
            'categories': {
                'harassment': Value('bool'),
                'harassment/threatening': Value('bool'),
                'harassment_threatening': Value('bool'),
                'hate': Value('bool'),
                'hate/threatening': Value('bool'),
                'hate_threatening': Value('bool'),
                'self-harm': Value('bool'),
                'self-harm/instructions': Value('bool'),
                'self-harm/intent': Value('bool'),
                'self_harm': Value('bool'),
                'self_harm_instructions': Value('bool'),
                'self_harm_intent': Value('bool'),
                'sexual': Value('bool'),
                'sexual/minors': Value('bool'),
                'sexual_minors': Value('bool'),
                'violence': Value('bool'),
                'violence/graphic': Value('bool'),
                'violence_graphic': Value('bool')
            },
            'category_scores': {
                'harassment': Value('float64'),
                'harassment/threatening': Value('float64'),
                'harassment_threatening': Value('float64'),
                'hate': Value('float64'),
                'hate/threatening': Value('float64'),
                'hate_threatening': Value('float64'),
                'self-harm': Value('float64'),
                'self-harm/instructions': Value('float64'),
                'self-harm/intent': Value('float64'),
                'self_harm': Value('float64'),
                'self_harm_instructions': Value('float64'),
                'self_harm_intent': Value('float64'),
                'sexual': Value('float64'),
                'sexual/minors': Value('float64'),
                'sexual_minors': Value('float64'),
                'violence': Value('float64'),
                'violence/graphic': Value('float64'),
                'violence_graphic': Value('float64')
            },
            'flagged': Value('bool')
        }],
        'detoxify_moderation': [{
            'identity_attack': Value('float64'),
            'insult': Value('float64'),
            'obscene': Value('float64'),
            'severe_toxicity': Value('float64'),
            'sexual_explicit': Value('float64'),
            'threat': Value('float64'),
            'toxicity': Value('float64')
        }],
        'toxic': Value('bool'),
        'redacted': Value('bool'),
        'state': Value('string'),
        'country': Value('string'),
        'hashed_ip': Value('string'),
        'header': {
            'accept-language': Value('string'),
            'user-agent': Value('string')
        },
        'model': Value('string')
    })

    return new_features

# Example usage:
def convert_dataset_types(dataset):
    """
    Convert the types in a dataset using the new features structure.
    """
    new_features = convert_header_types(dataset.features)
    
    # Cast the dataset to the new features
    converted_dataset = dataset.cast(new_features)
    
    return converted_dataset