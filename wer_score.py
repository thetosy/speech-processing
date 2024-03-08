import argparse
import wer

# function calls wer.string_edit_distance() on every utterance
# and accumulates the errors for the corpus. Then, report the word error rate (WER)
# and the sentence error rate (SER). The WER includes the the total errors as well as
# reporting the percentage of insertions, deletions and substitutions.
# The function signature is
# num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(ref=reference_string, hyp=hypothesis_string)
#
def score(ref_trn=None, hyp_trn=None):
    total_tokens = total_errors = total_deletes = total_inserts = total_subs = ser =  0
    with open(ref_trn) as ref_file:
        ref_utterances = ref_file.read().split('\n')
    with open(hyp_trn) as hyp_file:
        hyp_utterances = hyp_file.read().split('\n')

    if len(ref_utterances) != len(hyp_utterances):
        raise ValueError('Number of utterances in files do not match! Please fix')

    for ref, hyp in zip(ref_utterances, hyp_utterances):
        num_tokens, num_errors, num_deletions, num_insertions, num_substitutions = wer.string_edit_distance(
            ref=ref, hyp=hyp)
        total_tokens += num_tokens
        total_errors += num_errors
        total_deletes += num_deletions
        total_inserts += num_insertions
        total_subs += num_substitutions
        ser += 1 if ref != hyp else 0


    wer_rate = (total_errors / total_tokens) * 100
    ser_rate = (ser / len(ref_utterances)) * 100

    print(f"Word Error Rate (WER): {wer_rate:.2f}%")
    print(f"Sentence Error Rate (SER): {ser_rate:.2f}%")
    print(f"Insertions: {total_inserts}, Deletions: {total_deletes}, Substitutions: {total_subs}")

    return total_tokens, total_errors, total_deletes, total_inserts, total_subs


if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Evaluate ASR results.\n"
                                                 "Computes Word Error Rate and Sentence Error Rate")
    parser.add_argument('-ht', '--hyptrn', help='Hypothesized transcripts in TRN format', required=True, default=None)
    parser.add_argument('-rt', '--reftrn', help='Reference transcripts in TRN format', required=True, default=None)
    args = parser.parse_args()

    if args.reftrn is None or args.hyptrn is None:
        RuntimeError("Must specify reference trn and hypothesis trn files.")

    score(ref_trn=args.reftrn, hyp_trn=args.hyptrn)
