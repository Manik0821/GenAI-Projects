from app.models import ask_llm, make_model


def main():
    try:
        print("Raw client test:")
        print(ask_llm("Say hello in one sentence."))

        print("\nLangChain model init test:")
        llm = make_model()
        response = llm.invoke("Reply with: model setup successful")
        print(response.content)
    except Exception as e:
        print("Smoke test failed:", e)


if __name__ == "__main__":
    main()