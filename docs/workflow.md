# GitHub Flow

This project follows the standard GitHub Flow for collaborative development:

1. **Create a branch:**

   ```bash
   git checkout main
   git pull
   git checkout -b feature/your-feature-name
   ```
2. **Make changes and commit:**

   ```bash
   git add .
   git commit -m "Short description of the changes"
   ```
3. **Open a Pull Request:**

   ```bash
   git push -u origin feature/your-feature-name
   ```

   * Go to GitHub → Pull Requests → New Pull Request
   * Select your branch, create a PR
4. **Wait for review and CI:**
   Wait for automatic checks to complete and address any comments.
5. **Merge:**
   After approval, merge to main (usually via "Squash and merge").
6. **Delete the branch:**

   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```
