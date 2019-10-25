import React from "react";
import "./css/tachyons.css";
import "./css/custom.css";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      inputText: "",
      outputText: "",
      showResult: false
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);
  }

  handleChange(event) {
    event.preventDefault();
    this.setState({ inputText: event.target.value });
  }

  handleSubmit(event) {
    event.preventDefault();

    fetch(
      `https://labs.wellcomecollection.org/nerd?query_text=${encodeURIComponent(
        this.state.inputText
      )}
        `
    )
      .then(response => response.json())
      .then(data => this.setState({ outputText: data["response"] }));

    this.setState({ showResult: true });
  }

  render() {
    return (
      <div>
        <form onSubmit={this.handleSubmit}>
          <textarea
            type="text"
            id="comment"
            class="lh-copy h4 db border-box w-100 ba b--black-20 pa2 br2 mb2"
            placeholder="Type some text here and the neural network will try to guess who and what you're talking about"
            onChange={this.handleChange}
          />
          <input
            type="submit"
            className="f6 b no-underline br-pill ph3 pv2 mb2 dib white bg-black"
            value="Get nerdy"
          />
        </form>
        {this.state.showResult ? (
          <div
            className="mt2 lh-copy bg-light-gray pa2 br2"
            dangerouslySetInnerHTML={{ __html: this.state.outputText }}
          />
        ) : null}
      </div>
    );
  }
}

export default App;
