# API call to GPT-4

import os
import openai
import time
import tiktoken
import sys

# set API keys
openai.api_key = os.getenv("OPENAI_API_KEY")

# set names of files to save conversations and token counts and costs to
convo_path = "conversations"
if not os.path.exists(convo_path):
    os.makedirs(convo_path)

# ~~~~~~~~~~~~~~~~~~~~~~~ CLASSES ~~~~~~~~~~~~~~~~~~~~~~~#
class Conversation:
    """A conversation with the GPT engine.
       A conversation is a list of interactions."""
    
    def __init__(self, system_msg) -> None:
        # set the conversation file based on the time the conversation started
        self.convo_file = f"{convo_path}/{get_time('file')}"

        # set default settings
        self.continuing = False
        self.model = "gpt-3.5-turbo" # gpt-4
        self.max_tokens = 1000
        self.temperature = .5

        # set the conversation to an empty list
        self.messages = [{"role": "system", "content": system_msg}]

        # set the token count to 0
        self.token_count = 0
    
    def get_response(self):
        """Get a response from the GPT engine."""
        response = openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count
        )

        # get the response text
        response_text = response["choices"][0]["text"]

        # update token count
        self.token_count += response["usage"]["total_tokens"]

        return response
    
    def stream_response(self):
        """Stream a response from the GPT engine."""
        response_text = ""

        for chunk in openai.ChatCompletion.create(
            model = self.model,
            messages = self.messages,
            temperature = self.temperature,
            max_tokens = self.max_tokens - self.token_count,
            stream = True
        ):
            content_response = chunk["choices"][0]["delta"]
            content = content_response.get('content', '')
            if content is not None:
                 response_text += content
                 print(content, flush=True, end='')

        # update token count
        enc = tiktoken.encoding_for_model(self.model)
        self.token_count += len(enc.encode(response_text))
        self.token_count += len(enc.encode(self.messages[-1]["content"]))

        return response_text

    def converse(self):
        """Start a conversation with the GPT engine."""
        # start a conversation
        while True:

            print(self.token_count)

            # get the user's message
            user_msg = input("User: ")

            # add the user's message to the conversation
            self.messages.append({"role": "user", "content": user_msg})

            # get the response from the GPT engine
            response = self.stream_response()

            # add the response to the conversation
            self.messages.append({"role": "system", "content": response})

# ~~~~~~~~~~~~~~~~~~~~~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~#
def get_time(type = "text"):
    '''Get the current time in the format "dd/mm/yyyy hh:mm:ss" for text or "yyyymmdd_hhmmss" for file names.'''
    if type == "text":
        return time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
    elif type == "file":
        return time.strftime("%Y%m%d_%H%M%S", time.localtime())
    else:
        raise ValueError("type must be either 'text' or 'file'")

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~#
def main():
    """Main function."""
    # ask for a system message
    system_msg = input("System message: ")

    # create a new conversation
    C = Conversation(system_msg)

    try:
        C.converse()
    except Exception as e:
        print(e)

        # Print the line the error occurs on 
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(exc_type, fname, exc_tb.tb_lineno)
    except KeyboardInterrupt:
        sys.exit()

if __name__ == "__main__":
    main()