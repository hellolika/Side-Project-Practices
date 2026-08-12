CREATE PROCEDURE [dbo].[HRMS_ChangePassword.1.0.0]
	@memberId int,
	@oldPassword NVARCHAR(88),
	@newPassword NVARCHAR(88)
AS
BEGIN
	SET NOCOUNT ON;

	IF(@oldPassword = (SELECT [Password] FROM [dbo].[Member] WHERE [Id] = @memberId))
	BEGIN
		UPDATE [dbo].[Member]
		SET [Password] = @newPassword,
			[IsFirstTimeUser] = 0
		WHERE [Id] = @memberId
		SELECT 0 AS ErrorCode;
	END
	ELSE
		SELECT 18 AS ErrorCode;
END
