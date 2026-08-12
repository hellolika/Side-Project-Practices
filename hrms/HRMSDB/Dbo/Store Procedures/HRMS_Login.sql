CREATE PROCEDURE [dbo].[HRMS_Login.2.0.0]
	@email NVARCHAR(100),
	@password NVARCHAR(88)
AS
BEGIN 
	SET NOCOUNT ON;

	WITH TempTable AS (
		SELECT 0 AS [ErrorCode], [Id] AS [MemberId], [Username], [Email], [PhoneNumber], [Address],
		[Permission], [BankAccount], [IsInProbation],[BankName],[JoinDate], t.[TeamName],
		[Remark], m.[TeamId], [JobGrade], [Salary], [Position], [IsFirstTimeUser] FROM [dbo].[Member] m WITH(NOLOCK)
		INNER JOIN [dbo].[Team] t WITH(NOLOCK) ON t.[TeamId] = m.[TeamId]
		WHERE [Email] = @email AND [Password] = @password AND [IsResigned] = 0 AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL)
		UNION ALL
		SELECT 28 AS [ErrorCode], NULL, NULL, NULL, NULL,
		NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,NULL,NULL, NULL, NULL, NULL
	)

	SELECT TOP 1 [ErrorCode], [MemberId], [Username], [Email], [PhoneNumber], [Address],
	[Permission], [BankAccount], [IsInProbation], [BankName], [JoinDate],
	[Remark], [TeamId], [TeamName], [JobGrade], [Salary], [Position], [IsFirstTimeUser] FROM TempTable
	ORDER BY [ErrorCode] ASC

END

