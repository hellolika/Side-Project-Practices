CREATE PROCEDURE [dbo].[HRMS_ResetMemberPassword.1.0.0]
	@memberId INT,
	@password NVARCHAR(88)
AS
BEGIN
	SET NOCOUNT ON;
	IF Exists (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Id] = @memberId) 
	BEGIN
		UPDATE [dbo].[Member]
		SET [Password] = @password
		WHERE [Id] = @memberId
		SELECT 0 AS ErrorCode;
	END
	ELSE
		SELECT 3 AS ErrorCode
END
