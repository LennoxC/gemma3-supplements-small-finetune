# used to print the token label and id beside each token in the prompt.
# this was useful for debugging the masking, but is not used in production.
def visualize_masking(batch, processor, num_tokens=5000):
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    tokenizer = processor.tokenizer

    for i in range(input_ids.size(0)):
        seq_ids = input_ids[i].tolist()
        label_ids = labels[i].tolist()

        print(f"\nSequence {i}:")
        for j in range(min(len(seq_ids), num_tokens)):
            token = tokenizer.convert_ids_to_tokens(seq_ids[j])
            label = label_ids[j]
            display = f"{token}({label_ids[j]}, {seq_ids[j]})"
            print(f"{display}", end=" ")
        print("\n")