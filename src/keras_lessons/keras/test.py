from sentence_transformers import SentenceTransformer, util

# Загружаем предобученную модель
model = SentenceTransformer("all-MiniLM-L6-v2")


def evaluate_similarity(flowline_input, activity, flowline_output, generated_story):
    """
    Оценивает, насколько user story соответствует диаграмме состояний.
    """
    # Формируем тексты для сравнения
    diagram_text = f"{flowline_input}. {activity}. {flowline_output}."
    story_text = generated_story
    embeddings = model.encode([diagram_text, story_text], convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    return similarity_score


# Пример входных данных
flowline_input = "Order details confirmed"
activity = "Prepare ingredients"
flowline_output = "Ingredients ready"
generated_story = "As a baker, when order details are confirmed, I want to prepare ingredients, so that I can have the ingredients ready"

# Вычисляем схожесть
similarity = evaluate_similarity(
    flowline_input, activity, flowline_output, generated_story
)

print(f"Similarity Score: {similarity:.2f}")  # Чем ближе к 1, тем лучше
