lines = open('app.py', 'r', encoding='utf-8').readlines()

insert = '''
def _speak_chart(label: str) -> None:
    from voice import speak, autoplay_audio
    try:
        result = run_agent_with_trace(
            f"In 2 sentences, explain the key insight from this chart: {label}. Be specific with numbers."
        )
        text = result.get("content", "")
        if text:
            audio = speak(text)
            if audio:
                st.markdown(autoplay_audio(audio), unsafe_allow_html=True)
                st.caption(f"🔊 {text[:140]}")
    except Exception as e:
        st.caption(f"Voice error: {e}")


def voice_chart_button(label: str, key: str) -> None:
    if st.button("🔊 Explain", key=key, help="Hear Sage explain this chart"):
        with st.spinner("Sage is thinking..."):
            _speak_chart(label)

'''

# Insert before line 886 (0-indexed: 885)
insert_at = 885
lines.insert(insert_at, insert)
open('app.py', 'w', encoding='utf-8').writelines(lines)
print('Done')

# Verify
content = open('app.py', 'r', encoding='utf-8').read()
print('voice_chart_button found:', 'voice_chart_button' in content)
