import numpy as np
import argparse


class DataLoader:
    def __init__(self,file_path):
        self.file_path = file_path
        """Load The data and map the vectors according to words,
        """
        self.words,self.word_to_vec_map = self.read_glove_vecs()

               
    def read_glove_vecs(self):
        """Splitting of text and separating words and vectors and mapping them
        Args:
            file_path = text file directory, We use GloVe Vectors which provides much more information abou
            the meaning of individual words.

        Returns:
            words: returns the words of vectors.
            word_to_vec_map: Here, It maps the word and their vectors. 

        """
        with open(self.file_path,'r',encoding="utf8") as file:
            words = set()
            word_to_vec_map = dict()
            for lines in file:
                line = lines.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:],dtype = np.float64)
            print("Data Loaded Successfully")
            return words,word_to_vec_map

    def cosine_similarity(self,u,v):
        """To measure how similar two words are, we need to measure the degree of similarity between
        two embedding vectors of two words,
        
        If u and v are very similar, their cosine similarity will be close to 1.
        
        Arguments:
            u -- Embedding vectors of first word.
            v -- Embedding vectors of second word.

        Returns:
            cosine_similarity -- The cosine similarity between u and v defined by the formula
        """
        # dot product between u and v
        dot_pro = np.dot(u,v)

        #L2 norm of v
        norm_u = np.sqrt(np.sum(u*u))
        #L2 norm of v
        norm_v = np.sqrt(np.sum(v*v))

        # compute cosine similarity

        cosine_sim = dot_pro/(norm_u * norm_v)
        return cosine_sim

    def complete_analog(self,word_a,word_b,word_c,word_to_vec):
        word_a,word_b,word_c = word_a.lower(),word_b.lower(),word_c.lower()
        e_a,e_b,e_c = word_to_vec[word_a],word_to_vec[word_b],word_to_vec[word_c]

        words = word_to_vec.keys()
        max_cosine_sim = -100
        best_word = None

        for w in words:
            if w in [word_a,word_b,word_c]:
                continue
            cosine_simi = self.cosine_similarity(e_b-e_a,word_to_vec[w]-e_c)

            if cosine_simi > max_cosine_sim:
                max_cosine_sim = cosine_simi
                best_word = w

        return best_word









    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-tp", "--txt_path", default="glove.6B.50d.txt",help='Give Your Text File Directory.')
#     args = parser.parse_args()
#     if args.txt_path:
#         path = args.txt_path
#         load = DataLoader(path)
#     else:
#         raise "Give text file directory"



        
    
        