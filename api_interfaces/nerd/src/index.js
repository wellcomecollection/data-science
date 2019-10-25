import React from "react";
import ReactDOM from "react-dom";
import App from "./App";

class NERD extends React.Component {
  render() {
    return (
      <article className="center measure-wide ph5 helvetica">
        <div>
          <h1 className="f1 tracked-tight">
            <img className="pr2 h2" src="../icon.svg" title="hello!" alt=":)" />
            NERD
          </h1>
          <p className="lh-copy">
            Annotate text with subjects and entities using a Named Entity
            Recognition + Disambiguation (NERD) model, powered by Wikipedia,
            Wikidata, and{" "}
            <span role="img" aria-label="sparkle">
              ✨
            </span>
            machine learning
            <span role="img" aria-label="sparkle">
              ✨
            </span>
          </p>
        </div>
        <App />
      </article>
    );
  }
}

ReactDOM.render(<NERD />, document.getElementById("root"));