class Chunking:

    def chunk_text(self, text: str, chunk_size: int = 300, overlap: int = 75) -> list[str]:

        if overlap >= chunk_size:
            raise ValueError("Overlap cannot be larger than or equal to chunk size")
        
        if overlap < 0:
            raise ValueError("Overlap cannot be negative")

        # Only removes white spaces from the beginning and end,
        # so it preserves paragraph/newline structure for recursive splitting.
        text = text.strip()

        if not text:
            return []
        
        # Split raw chunks smaller than chunk_size so that adding overlap
        # does not make the final chunks go above chunk_size.
        target_chunk_size = chunk_size - overlap

        # Return the raw chunks after recursivley splitting,
        # trying first paragraphs, then sentences, then newlines, then words.
        chunks = self._recursive_split(
            text=text,
            chunk_size = target_chunk_size,
            separators =["\n\n", ". ", "\n"]
        )

        # Add overlap to chunks
        chunks_with_overlap = self._add_overlap(chunks, overlap)

        return chunks_with_overlap

    def _recursive_split(self, text: str, chunk_size: int, separators: list[str]) -> list[str]:

        # Base case 1: Text is already small enough. If the text is already short enough, do not split it.
        if self._word_count(text) <= chunk_size:
            return [text]
        
        # Base case 2: We have tried seperating by paragraphs, sentences, and newlines, so just split by words.
        if not separators:
            return self._split_by_words(text, chunk_size)
        
        # Take the seperator at the front of the list
        separator = separators[0]

        # Remaining seperators
        remaining_separators = separators[1:]

        # split the text into pieces according to the seperator.
        pieces = text.split(separator)

       # If splitting by the current separator did not actually split the text, then try the next separator instead.
        if len(pieces) == 1:
            return self._recursive_split(text, chunk_size, remaining_separators)

        chunks = []
        current_chunk = ""

        for index, piece in enumerate(pieces):

            # Each individual piece may still have whitespace at its own start or end, so remove it.
            piece = piece.strip()

            # if piece is an empty string, "", go to the next piece
            if not piece:
                continue
            
            # If we split by ". ", the separator was removed from every sentence
            # except the final piece, so add the full stop back to non-final pieces.
            if separator == ". " and index < len(pieces) - 1:
                piece = piece + "."

            # if current chunk is non empty, add a space and add piece to it.
            if current_chunk:
                candidate_chunk = current_chunk + " " + piece
            else:
                candidate_chunk = piece # If current chunk is empty, "", its just equal to our first piece.

            # If adding this piece to the current chunk still keeps us under the allowed chunk size, then accept it.
            if self._word_count(candidate_chunk) <= chunk_size:
                current_chunk = candidate_chunk

            else:
                # If current_chunk is non-empty and adding the next piece would go over chunk_size, then add current chunk into chunks.
                if current_chunk:
                    chunks.append(current_chunk)

                # If the piece itself is too large, split it further using smaller separators.
                if self._word_count(piece) > chunk_size:
                    smaller_chunks = self._recursive_split(
                        text=piece,
                        chunk_size=chunk_size,
                        separators=remaining_separators
                    )
                    chunks.extend(smaller_chunks)

                    # We have handled this oversized piece by recursively splitting it,
                    # so there is no current chunk left to build.
                    current_chunk = ""

                # Otherwise, the piece is small enough to become the start of the next chunk.
                else:
                    current_chunk = piece


        # After the loop, save the final chunk if we were still building one.
        # current_chunk may be empty if the last piece was too large and was already
        # handled by recursion, so only append it if it was still being built.
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _add_overlap(self, chunks: list[str], overlap: int) -> list[str]:

        # If overlap is 0, we do not have to change chunks, so just return.
        if overlap == 0:
            return chunks

        chunks_with_overlap = []

        for index, chunk in enumerate(chunks):

            # The first chunk just gets added without an overlap.
            if index == 0:
                chunks_with_overlap.append(chunk)
                continue
            
            # split the previous chunk, i - 1, into a list of words
            previous_chunk_words = chunks[index - 1].split()

            # Take the last `overlap` words from the chunk i - 1 to use as context.
            overlap_words = previous_chunk_words[-overlap:]

            # takes the retrieved overlap_words from chunk i - 1 and joins them into a string
            overlap_text = " ".join(overlap_words)

            # Prepend the overlap text to the start of chunk i.
            chunk_with_overlap = f"{overlap_text} {chunk}"

            # Add it to the chunks_with_overlap list
            chunks_with_overlap.append(chunk_with_overlap)

        return chunks_with_overlap

    # Helper fucntion to handle the last resort for recursion
    def _split_by_words(self, text: str, chunk_size: int) -> list[str]:

        words = text.split()
        chunks = []

        for start in range(0, len(words), chunk_size):
            chunk = " ".join(words[start:start + chunk_size])
            chunks.append(chunk)

        return chunks

    # Helper function so we don't have to keep writing len(text.split())
    def _word_count(self, text: str) -> int:

        return len(text.split())
