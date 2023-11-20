# app.py
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
from bs4 import BeautifulSoup
import requests
from hugchat import hugchat
from hugchat.login import Login
import mediapy as mp
from better_profanity import Profanity
from diffusers import EulerAncestralDiscreteScheduler as EAD
from diffusers import StableDiffusionPipeline as sdp

# Initialize the question-answering pipeline
qa_model = pipeline("question-answering")

def has_profanity(text):
    return Profanity().contains_profanity(text)

def filter_text(text):
    while has_profanity(text):
        text = input("Please provide an alternative prompt: ")
    return text
def generate_hook(prompt, model="gpt2", style="story", grade_level="primary", tone="humorous"):
    # Customize the prompt based on selected style, grade level, and tone
    prompt = f"Create a {style.lower()} hook for {grade_level} students with a {tone.lower()} tone about {prompt}"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    input_ids = tokenizer.encode(prompt, return_tensors="pt", max_length=200, truncation=True)
    output = model.generate(input_ids, max_length=150, temperature=0.7, num_beams=5, no_repeat_ngram_size=2)

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def images():
      model = "dreamlike-art/dreamlike-photoreal-2.0"
      scheduler = EAD.from_pretrained(model, subfolder="scheduler")

      pipe = sdp.from_pretrained(
          model,
          scheduler=scheduler
          )
      device = "cuda"

      pipe = pipe.to(device)
      prompt = "baking a cake"
      num_images=3
      filtered_input = filter_text(prompt)
      images = pipe(
      filtered_input,
      height = 512,
      width = 512,
      num_inference_steps = 30, #more no of steps,  better results
      guidance_scale = 9, #more no of steps,  better results
      num_images_per_prompt = num_images

      ).images
      st.image(images)

def hfgpt():
  st.write("logging in")

  email='ahmedmuzammil.ai@gmail.com'
  passwd='Teamsmc12#'
  sign = Login(email, passwd)
  cookies = sign.login()
  st.write("logging into hugchat api")
  # Save cookies to the local directory
  cookie_path_dir = "./cookies_snapshot"
  sign.saveCookiesToDir(cookie_path_dir)

  # Load cookies when you restart your program:
  # sign = login(email, None)
  # cookies = sign.loadCookiesFromDir(cookie_path_dir) # This will detect if the JSON file exists, return cookies if it does and raise an Exception if it's not.

  # Create a ChatBot
  chatbot = hugchat.ChatBot(cookies=cookies.get_dict())  # or cookie_path="usercookies/<email>.json"

  # st.write("querying")

  # non stream response
  query_result = chatbot.query("tell me something about cop28")
  print(query_result) # or query_result.text or query_result["text"]
  st.write(query_result.text)

  # stream response
  for resp in chatbot.query(
      "Hello",
      stream=True
  ):
      print(resp)
  st.write("using web search")

  # Use web search *new
  query_result = chatbot.query("give me a hook related to fractions concept in math", web_search=True)
  print(query_result) # or query_result.text or query_result["text"]
  for source in query_result.web_search_sources:
      print(source.link)
      print(source.title)
      print(source.hostname)

  # Create a new conversation
  id = chatbot.new_conversation()
  chatbot.change_conversation(id)

  # Get conversation list
  conversation_list = chatbot.get_conversation_list()

  # Switch model (default: meta-llama/Llama-2-70b-chat-hf. )
  chatbot.switch_llm(0) # Switch to `OpenAssistant/oasst-sft-6-llama-30b-xor`
  chatbot.switch_llm(1)
  return query_result

def web_scrape_example(topic):
    url = f'https://en.wikipedia.org/wiki/{topic}'
    response = requests.get(url)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the main content
        content_div = soup.find('div', {'id': 'mw-content-text'})
        if content_div:
            paragraphs = content_div.find_all('p')  # Extracting paragraphs
            content = '\n'.join([p.get_text() for p in paragraphs])
            return content
        else:
            return "No content found on the Wikipedia page."
    else:
        return "Failed to fetch content. Check your internet connection or try a different topic."

def main():
    st.title("Teacher's Hook Generator App")

    # Sidebar with three vertical sections
    with st.sidebar:
        st.markdown("## Style")
        style_selected = st.radio("", ["Story", "Question", "Image", "Video", "Real Event", "Surprising Fact"])

        st.markdown("## Grade Level")
        grade_level_selected = st.radio("", ["Primary", "Secondary"])

        st.markdown("## Tone")
        tone_selected = st.radio("", ["Humorous", "Serious"])

    # Collect user inputs
    topic = st.text_input("Enter the topic:")

    # Main section
    if st.button("Generate Hook") and topic:
        generated_hook_gpt2 = generate_hook(topic, style=style_selected, grade_level=grade_level_selected, tone=tone_selected)
        st.subheader("Generated Hugging Face Hook:")
        st.write(generated_hook_gpt2)

    if st.button("Web Scrape Example") and topic:
        scraped_text = web_scrape_example(topic)
        st.write("Web Scraped Text:")
        st.write(scraped_text)


        # Allow the user to ask queries on the scraped data
        question = st.text_input("Ask a question about the scraped data:")
        if question:
            answer = qa_model(question=question, context=scraped_text)
            st.write("Answer:")
            st.write(answer['answer'])
    if style_selected=="Real Event":
      st.write("real event here")
      st.write("scraping")
      scraped=hfgpt()
      st.write(scraped.text)
      st.write("scraped")
    if style_selected=="Image":
      images()

      
if __name__ == "__main__":
    main()
