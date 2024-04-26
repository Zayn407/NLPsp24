from sample_code_1 import score_length
from sample_code_1 import score_spelling
from sample_code_2 import score_subject_verb_agreement
from sample_code_2 import score_verb_usage

def main():
    while True:
        # Prompt the user to input the name of the essay file or type 'quit' to exit
        essay_filename = input("Please enter the name of the essay file (including .txt extension) or type 'quit' to exit: ")

        if essay_filename.lower() == 'quit':
            break

        # topic = input("Please enter the topic of the essay: ")

        # Read the essay file
        try:
            with open(essay_filename, 'r') as file:
                essay = file.read()
        except Exception as e:
            print(f"Error reading the file: {e}")
            continue  # Skip to the next iteration if there is an error

        # Calculate and print the scores
        print("\nCalculating scores...")
        # Calculate each score
        a = score_length(essay)
        b = score_spelling(essay)  # Note that this score is on a 0 to 4 scale
        ci = score_subject_verb_agreement(essay)
        cii = score_verb_usage(essay)


        # Print the scores
        print(f"\nLength score (a): {a}")
        print(f"Spelling score (b): {b} (on a scale of 0 to 4)")
        print(f"Subject-Verb agreement score (c.i): {ci}")
        print(f"Verb tense usage score (c.ii): {cii}")


if __name__ == "__main__":
    main()