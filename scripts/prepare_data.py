#!/usr/bin/env python3
"""
Download and prepare training data for ActuallyOpenAI.

Uses high-quality datasets from Hugging Face.
"""

import os
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_training_data(output_path: str = "data/train.txt", size_mb: int = 10):
    """Download training data from various sources."""
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    print("Downloading training data...")
    
    try:
        from datasets import load_dataset
        
        # Load a mix of datasets
        texts = []
        
        # 1. OpenAssistant conversations (high quality chat data)
        print("Loading OpenAssistant data...")
        try:
            ds = load_dataset("OpenAssistant/oasst1", split="train[:1000]")
            for item in ds:
                if item.get('text'):
                    texts.append(f"User: {item['text']}\n")
        except:
            print("Could not load OpenAssistant data")
        
        # 2. Tiny stories (good for language learning)
        print("Loading TinyStories data...")
        try:
            ds = load_dataset("roneneldan/TinyStories", split="train[:2000]")
            for item in ds:
                if item.get('text'):
                    texts.append(item['text'] + "\n\n")
        except:
            print("Could not load TinyStories data")
        
        # 3. Wikipedia (factual knowledge)
        print("Loading Wikipedia sample...")
        try:
            ds = load_dataset("wikipedia", "20220301.en", split="train[:500]")
            for item in ds:
                if item.get('text') and len(item['text']) > 100:
                    texts.append(item['text'][:2000] + "\n\n")
        except:
            print("Could not load Wikipedia data")
        
        # Write to file
        print(f"Writing {len(texts)} samples to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(texts))
        
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Saved {file_size:.1f} MB of training data")
        
        return output_path
        
    except ImportError:
        print("datasets library not found. Using built-in data.")
        return create_builtin_data(output_path)


def create_builtin_data(output_path: str = "data/train.txt") -> str:
    """Create built-in training data without external dependencies."""
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # High-quality conversational training data
    conversations = [
        # Greetings
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there!", "Hi! What would you like to know?"),
        ("Good morning!", "Good morning! How can I assist you?"),
        ("Hey", "Hey! What's on your mind?"),
        
        # AI/ML questions
        ("What is artificial intelligence?", "Artificial intelligence (AI) refers to the simulation of human intelligence in machines. These systems can learn from experience, adjust to new inputs, and perform tasks that typically require human intelligence like visual perception, speech recognition, decision-making, and language translation."),
        ("Explain machine learning", "Machine learning is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data, learn from it, and make predictions or decisions. Common types include supervised learning, unsupervised learning, and reinforcement learning."),
        ("What is deep learning?", "Deep learning is a subset of machine learning based on artificial neural networks with multiple layers. These networks can learn complex patterns from large amounts of data. Deep learning powers many modern AI applications including image recognition, natural language processing, and autonomous vehicles."),
        ("How do neural networks work?", "Neural networks are computing systems inspired by biological neurons. They consist of layers of interconnected nodes. Input data passes through these layers, with each node applying weights and activation functions. Through training, the network adjusts its weights to minimize prediction errors and learn patterns in the data."),
        ("What is a transformer model?", "Transformers are a type of neural network architecture introduced in the paper 'Attention is All You Need'. They use self-attention mechanisms to process sequential data in parallel, making them highly efficient. Transformers are the foundation of modern language models like GPT, BERT, and our ActuallyOpenAI model."),
        
        # Programming
        ("What is Python?", "Python is a high-level, interpreted programming language known for its simplicity and readability. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python is widely used in web development, data science, AI, automation, and scientific computing."),
        ("How do I learn programming?", "Start with the basics: choose a beginner-friendly language like Python, learn variables, data types, and control flow. Practice regularly with small projects. Use online resources like tutorials, documentation, and coding challenges. Build projects that interest you, and don't be afraid to make mistakes - that's how you learn!"),
        ("What is an API?", "An API (Application Programming Interface) is a set of protocols and tools that allows different software applications to communicate. It defines how components should interact, enabling developers to access functionality from other services without knowing their internal implementation. APIs power most modern web and mobile applications."),
        
        # General knowledge
        ("What is the speed of light?", "The speed of light in a vacuum is approximately 299,792,458 meters per second (about 186,282 miles per second). This is the maximum speed at which information or matter can travel in the universe according to Einstein's theory of special relativity."),
        ("Tell me about the solar system", "Our solar system consists of the Sun and everything gravitationally bound to it. This includes eight planets (Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune), dwarf planets like Pluto, moons, asteroids, comets, and other objects. The Sun contains 99.86% of the system's mass."),
        ("What is climate change?", "Climate change refers to long-term shifts in global temperatures and weather patterns. While natural factors can cause climate variations, human activities since the Industrial Revolution, particularly burning fossil fuels, have been the main driver of recent warming. Effects include rising sea levels, extreme weather, and ecosystem disruption."),
        
        # Helpful responses
        ("Thank you", "You're welcome! Is there anything else I can help you with?"),
        ("That's helpful", "I'm glad I could help! Feel free to ask if you have more questions."),
        ("I don't understand", "Let me explain it differently. What specific part would you like me to clarify?"),
        ("Can you help me?", "Of course! I'm here to help. What do you need assistance with?"),
        ("Tell me more", "I'd be happy to elaborate. What specific aspect would you like to learn more about?"),
        
        # ActuallyOpenAI specific
        ("What is ActuallyOpenAI?", "ActuallyOpenAI is a decentralized AI training platform where anyone can contribute their computing power and earn rewards. Unlike centralized AI companies, ActuallyOpenAI is community-owned - contributors earn AOAI tokens and share in the revenue generated by the AI services."),
        ("How do I earn AOAI tokens?", "You can earn AOAI tokens by running a worker node that contributes GPU or CPU power to train AI models. The more compute you contribute, and the better your reputation score, the more tokens you earn. These tokens entitle you to a share of the platform's revenue."),
        ("How is ActuallyOpenAI different?", "ActuallyOpenAI is unique because it's truly decentralized and community-owned. Instead of a single company controlling AI development, contributors worldwide collectively train models and share in the benefits. All code is open source, and revenue is distributed as dividends to token holders."),
    ]
    
    # Convert to training format
    texts = []
    for user_msg, assistant_msg in conversations:
        texts.append(f"User: {user_msg}\nAssistant: {assistant_msg}\n")
    
    # Repeat to create more training data
    full_text = "\n".join(texts * 100)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Created {file_size:.2f} MB of training data at {output_path}")
    
    return output_path


if __name__ == "__main__":
    output = create_builtin_data()
    print(f"Training data ready at: {output}")
