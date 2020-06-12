import React, { Component } from "react";
import Header from "./header";
import Choice from "./choice";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  componentDidMount() {
    fetch("/data_for_interface")
      .then((response) => response.json())
      .then((data) => this.setState(data));
  }

  render() {
    return (
      <div>
        <Header query_id={this.state.query_id}></Header>
        <div>
          <Choice
            className="center"
            choice_name="A"
            index={this.state.index_a}
            alt_index={this.state.index_b}
            image_ids={this.state.similar_image_ids_a}
          ></Choice>
          <Choice
            className="center"
            choice_name="B"
            index={this.state.index_b}
            alt_index={this.state.index_a}
            image_ids={this.state.similar_image_ids_b}
          ></Choice>
        </div>
      </div>
    );
  }
}

export default App;
