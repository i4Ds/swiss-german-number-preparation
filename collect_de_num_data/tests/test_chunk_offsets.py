"""Small merge tests; GPU dependencies are stubbed and never imported."""
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_PATH = REPOSITORY_ROOT / 'collect_de_num_data/src/labeling/llm_inference.py'
SPEC = importlib.util.spec_from_file_location('chunk_offset_inference', INFERENCE_PATH)
INFERENCE = importlib.util.module_from_spec(SPEC)
# Merge logic needs only the standard library. Do not initialize torch or vLLM
# when running these tests on a login node.
VLLM_STUB = types.ModuleType('vllm')
VLLM_STUB.LLM = object
VLLM_STUB.SamplingParams = object
with patch.dict(sys.modules, {
    'torch': types.ModuleType('torch'),
    'yaml': types.ModuleType('yaml'),
    'vllm': VLLM_STUB,
}):
    SPEC.loader.exec_module(INFERENCE)

PREPROCESS_PATH = REPOSITORY_ROOT / 'preprocess_num_of_datasets/scripts/preprocess_dataset_numbers.py'
PREPROCESS_SPEC = importlib.util.spec_from_file_location('chunk_offset_preprocess', PREPROCESS_PATH)
PREPROCESS = importlib.util.module_from_spec(PREPROCESS_SPEC)
sys.modules[PREPROCESS_SPEC.name] = PREPROCESS
PREPROCESS_SPEC.loader.exec_module(PREPROCESS)


def response(text, number=None, *, quality='Good'):
    annotations = []
    if number is not None:
        start = text.index(number)
        annotations.append({
            'number': number, 'start': start, 'end': start + len(number),
            'type': 'NUMBER_CARDINAL', 'value': number,
        })
    return {'text': text, 'text_quality': quality, 'numbers': annotations}


def merge(responses, sources=None):
    return json.loads(INFERENCE.merge_chunk_responses(
        [json.dumps(item, ensure_ascii=False) for item in responses], sources))


class ChunkOffsetTest(unittest.TestCase):
    def test_shifted_offsets_pass_downstream_conversion(self):
        chunks = ['Zahl 12. ', 'Zahl 34.']
        result = merge([response(chunks[0], '12'), response(chunks[1], '34')], chunks)
        self.assertEqual([(n['start'], n['end']) for n in result['numbers']],
                         [(5, 7), (14, 16)])
        converted, _ = PREPROCESS.apply_annotations(
            ''.join(chunks), result, dialect='ch_sg',
            converter=lambda info, dialect: {'12': 'zwölf', '34': 'vieredrissig'}[info['number']],
        )
        self.assertEqual(converted, 'Zahl zwölf. Zahl vieredrissig.')

    def test_repeated_numbers_preserve_their_occurrences_and_metadata(self):
        chunks = ['12 ', '12 ', '12']
        responses = [response(chunk, '12') for chunk in chunks]
        for i, item in enumerate(responses):
            item['numbers'][0]['subtype'] = f'occurrence-{i}'
        result = merge(responses, chunks)
        self.assertEqual([n['start'] for n in result['numbers']], [0, 3, 6])
        self.assertEqual([n['subtype'] for n in result['numbers']],
                         ['occurrence-0', 'occurrence-1', 'occurrence-2'])

    def test_unicode_whitespace_and_unannotated_chunks_count_as_characters(self):
        chunks = ['Ä 12.\n', '\tü🙂 ', ' 34.']
        result = merge([response(chunks[0], '12'), response(chunks[1]),
                        response(chunks[2], '34', quality='Okay')], chunks)
        self.assertEqual(result['text'], 'Ä 12.\n\tü🙂  34.')
        self.assertEqual(result['numbers'][1]['start'], 11)
        self.assertEqual(result['numbers'][1]['end'], 13)
        self.assertEqual(result['text_quality'], 'Okay')

    def test_uses_original_chunk_lengths_when_response_text_is_rewritten(self):
        chunks = ['Zahl 12. ', 'Zahl 34.']
        responses = [response(chunk, number) for chunk, number in zip(chunks, ['12', '34'])]
        responses[0]['text'] = 'A changed answer with a different length'
        result = merge(responses, chunks)
        self.assertEqual(result['text'], ''.join(chunks))
        self.assertEqual(result['numbers'][1]['start'], 14)

    def test_optional_source_texts_use_response_texts(self):
        result = merge([response('12 ', '12'), response('34', '34')])
        self.assertEqual(result['text'], '12 34')
        self.assertEqual(result['numbers'][1]['start'], 3)

    def test_invalid_local_offsets_are_rejected_before_shifting(self):
        for start, end in [(None, 7), (5, None), ('5', 7), (True, 7),
                           (-1, 7), (5, 5), (5, 100), (14, 16)]:
            with self.subTest(start=start, end=end):
                invalid = response('Zahl 34.', '34')
                invalid['numbers'][0].update(start=start, end=end)
                with self.assertRaisesRegex(ValueError, 'Chunk 2.*invalid chunk-local offsets'):
                    merge([response('Zahl 12. ', '12'), invalid], ['Zahl 12. ', 'Zahl 34.'])

    def test_local_offsets_must_select_number_in_original_source(self):
        invalid = response('Zahl 34.', '34')
        with self.assertRaisesRegex(ValueError, 'offsets do not select number'):
            merge([invalid], ['Zahl 12.'])

    def test_missing_offsets_are_rejected(self):
        invalid = response('12', '12')
        del invalid['numbers'][0]['start']
        del invalid['numbers'][0]['end']
        with self.assertRaisesRegex(ValueError, 'invalid chunk-local offsets'):
            merge([invalid], ['12'])

    def test_malformed_annotation_is_rejected(self):
        invalid = response('12')
        invalid['numbers'] = [None]
        with self.assertRaisesRegex(ValueError, 'annotation 1 must be an object'):
            merge([invalid], ['12'])

    def test_mismatched_chunk_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'chunk counts differ'):
            merge([response('12', '12')], ['12', '34'])


if __name__ == '__main__':
    unittest.main()
