import React, { Component } from "react";

class Button extends Component {
  constructor(props) {
    super(props);
    this.handleClick = this.handleClick.bind(this);
  }

  handleClick() {
    // post the vote to the ES store
    fetch("/assessment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate_a: this.props.index,
        candidate_b: this.props.alt_index,
        winner: this.props.index,
      }),
    });

    window.location.reload();
  }

  render() {
    let button;
    if (this.props.index) {
      button = (
        <button
          onClick={this.handleClick}
          className="f5 no-underline black bg-animate hover-bg-black hover-white inline-flex items-center ph3 pv2 ba border-box mr4"
        >
          Vote {this.props.index}
        </button>
      );
    }
    return <div>{button}</div>;
  }
}

export default Button;
